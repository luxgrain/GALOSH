/* galosh_yuv_vk.c — GALOSH-YUV Vulkan host (FP16-storage fast engine).
 *
 * EN: Mirrors the OpenCL YUV host (galosh_yuv_gpu.c, default O-variant
 *     path) per YUV_BLUEPRINT.md: 13 kernels, Laplacian-MAD blind noise
 *     est (exact serial quickselect), GAT + o32 pass12 LOSH (shader
 *     REUSED from the RAW engine) + Makitalo LUT inverse + Y-guided
 *     bilateral LOESS chroma.  FP16 inter-phase storage contract v1
 *     (Y_stab / Y_den / guide snapshot / Cb / Cr planes are binary16
 *     with explicit RNE stores; compute stays f32).
 *     Proven deviations vs the OpenCL host (documented, metric-gated):
 *       - zero mid-frame readback (o32_build_inv_lut reads alpha/sigma^2
 *         from params[13]/[14] on device — blueprint §5.2);
 *       - the y_den -> y_stab copy-back is skipped (denorm + makitalo
 *         read y_den directly — same math, one copy less);
 *       - chroma_derive (dead compute, §5.6) is not dispatched.
 *     GALOSH-420 planar front-end: same composition as the OpenCL /
 *     CPU drivers via the SHARED header galosh_yuv420.h; core runs
 *     twice (full-res luma-only gray pass + native-lattice chroma pass)
 *     on ONE device init (blueprint §7).
 * JP: OpenCL YUV ホストの Vulkan ミラー（FP16 ストレージ契約 v1）。
 *     中間 readback ゼロ化・コピー 1 本削減・dead kernel 省略は
 *     文書化済みの proven deviation。420 は共有ヘッダで CPU/OpenCL と
 *     語彙・位相の乖離が構造的に不可能。
 *
 * CLI: galosh_yuv_vk.exe in.bin out.bin W H [s_y] [s_c]
 *        [--pix=420|422|444|400] [--depth=8..16] [--range=full|limited]
 *        [--matrix=bt601|bt709|bt2020|custom:Kr,Kb]
 *        [--eotf=srgb|g22|g24|bt709|hlg|pq|linear]
 *        [--siting=center|left|topleft] [--selftest-phase]
 * Env: GALOSH_VK_DEVICE, GALOSH_VERBOSE, GALOSH_SG
 * Build: build_vk.sh (glslc shaders + gcc -lvulkan-1).
 * (code: Apache-2.0)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "../galosh_yuv420.h"

#define CHECK(x) do { VkResult _r = (x); if(_r != VK_SUCCESS) { \
  fprintf(stderr, "VK error %d at %s:%d\n", _r, __FILE__, __LINE__); exit(1); } } while(0)

#define PARAMS_SIZE   32
#define GAT_LUT_SIZE  4096
#define O32_TILE      28
#define P_ALPHA       13
#define P_SIGMA_SQ    14
#define P_SIGMA_Y     20   /* blind sigma_lin (YGQ2a out) */
#define P_SIGMA_GAT   21   /* repurposed slot: sigma_gat (YG4a' out) */

static int g_verbose = 0;

/* ================= context (init once, reused across core runs) ===== */
static VkInstance       g_inst;
static VkPhysicalDevice g_pd;
static VkDevice         g_dev;
static VkQueue          g_q;
static uint32_t         g_qi;
static VkCommandPool    g_pool;
static VkPhysicalDeviceProperties g_props;
static VkDescriptorPool g_dpool;
static VkQueryPool      g_qpool;
static uint32_t         g_ts_n = 0;
static const char      *g_ts_label[256];
static int              g_sg_ok = 0;

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; } Buf;

static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_pd, &mp);
  for(uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  fprintf(stderr, "no memtype 0x%x\n", want); exit(1);
}

static Buf mkbuf(VkDeviceSize sz, VkBufferUsageFlags use, VkMemoryPropertyFlags props)
{
  Buf b; b.size = sz;
  VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = sz, .usage = use,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
  CHECK(vkCreateBuffer(g_dev, &bi, NULL, &b.buf));
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev, b.buf, &mr);
  VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size,
                               .memoryTypeIndex = find_mem(mr.memoryTypeBits, props) };
  CHECK(vkAllocateMemory(g_dev, &mai, NULL, &b.mem));
  CHECK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));
  return b;
}
#define DEVBUF(sz) mkbuf((sz), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | \
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, \
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

static void freebuf(Buf *b)
{
  if(b->buf) vkDestroyBuffer(g_dev, b->buf, NULL);
  if(b->mem) vkFreeMemory(g_dev, b->mem, NULL);
  b->buf = VK_NULL_HANDLE; b->mem = VK_NULL_HANDLE;
}

/* ================= pipelines ================= */
typedef struct {
  const char     *spv;
  int             nbind;
  int             push;
  int             sg32;
  VkDescriptorSetLayout dsl;
  VkPipelineLayout      pl;
  VkPipeline            pipe;
} Kern;

enum {
  K_SRGB2YCC, K_LAP_MAD, K_LAP_MAD_H16, K_SYNTH, K_GAT_FWD,
  K_NORM, K_DENORM, K_LUT_BUILD, K_LUT_FIN, K_PASS12, K_PASS12_SG,
  K_MAKITALO, K_LOESS, K_YCC2SRGB,
  K_COUNT
};
static Kern g_k[K_COUNT] = {
  [K_SRGB2YCC]   = { "yuv_srgb2ycc",      4,  4 },
  [K_LAP_MAD]    = { "yuv_lap_mad",       3, 16 },
  [K_LAP_MAD_H16]= { "yuv_lap_mad_h16",   3, 16 },
  [K_SYNTH]      = { "yuv_synth_alpha",   1, 12 },
  [K_GAT_FWD]    = { "yuv_gat_fwd",       3,  4 },
  [K_NORM]       = { "yuv_sigma_norm",    2,  8 },
  [K_DENORM]     = { "yuv_sigma_denorm",  2,  8 },
  /* REUSED AS-IS from the RAW o32 engine (YUV_BLUEPRINT.md §6): the LUT
   * pair reads alpha/sigma^2 from params[13]/[14] on device (kills the
   * OpenCL mid-frame readback); pass12 is the proven LOSH kernel. */
  [K_LUT_BUILD]  = { "o32_build_inv_lut", 4,  0 },
  [K_LUT_FIN]    = { "o32_lut_finalize",  2,  0 },
  [K_PASS12]     = { "o32_pass12",        2, 16 },
  [K_PASS12_SG]  = { "o32_pass12_sg",     2, 16, .sg32 = 1 },
  [K_MAKITALO]   = { "yuv_makitalo",      5,  4 },
  [K_LOESS]      = { "yuv_loess",         5, 12 },
  [K_YCC2SRGB]   = { "yuv_ycc2srgb",      4,  4 },
};

static char g_exe_dir[1024];

static VkShaderModule load_spv(const char *name)
{
  char path[1200];
  snprintf(path, sizeof(path), "%s/shaders/%s.spv", g_exe_dir, name);
  FILE *f = fopen(path, "rb");
  if(!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  uint32_t *code = malloc(sz);
  if(fread(code, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", path); exit(1); }
  fclose(f);
  VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = (size_t)sz, .pCode = code };
  VkShaderModule m; CHECK(vkCreateShaderModule(g_dev, &ci, NULL, &m));
  free(code); return m;
}

static void make_kernel(Kern *k)
{
  VkDescriptorSetLayoutBinding binds[16];
  for(int i = 0; i < k->nbind; i++)
    binds[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
  VkDescriptorSetLayoutCreateInfo dsli = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = (uint32_t)k->nbind, .pBindings = binds };
  CHECK(vkCreateDescriptorSetLayout(g_dev, &dsli, NULL, &k->dsl));
  VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                              .offset = 0, .size = (uint32_t)(k->push ? k->push : 4) };
  VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1, .pSetLayouts = &k->dsl,
    .pushConstantRangeCount = k->push ? 1u : 0u, .pPushConstantRanges = &pcr };
  CHECK(vkCreatePipelineLayout(g_dev, &pli, NULL, &k->pl));
  VkShaderModule sm = load_spv(k->spv);
  VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT,
    .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
    .layout = k->pl };
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    .requiredSubgroupSize = 32 };
  if(k->sg32)
  {
    cpi.stage.pNext = &rss;
    cpi.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
  }
  CHECK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, NULL, &k->pipe));
  vkDestroyShaderModule(g_dev, sm, NULL);
}

static VkDescriptorSet mkset(int kid, const Buf *bufs[], int n)
{
  if(n != g_k[kid].nbind)
  { fprintf(stderr, "mkset %s: %d bufs != %d binds\n", g_k[kid].spv, n, g_k[kid].nbind); exit(1); }
  VkDescriptorSetAllocateInfo a = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = g_dpool, .descriptorSetCount = 1, .pSetLayouts = &g_k[kid].dsl };
  VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(g_dev, &a, &ds));
  VkDescriptorBufferInfo bi[16];
  VkWriteDescriptorSet wr[16];
  for(int i = 0; i < n; i++)
  {
    bi[i] = (VkDescriptorBufferInfo){ .buffer = bufs[i]->buf, .range = VK_WHOLE_SIZE };
    wr[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds, .dstBinding = (uint32_t)i, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i] };
  }
  vkUpdateDescriptorSets(g_dev, (uint32_t)n, wr, 0, NULL);
  return ds;
}

/* ================= command recording ================= */
static VkCommandBuffer cb_begin(void)
{
  VkCommandBufferAllocateInfo a = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = g_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
  VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(g_dev, &a, &cb));
  VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
  CHECK(vkBeginCommandBuffer(cb, &bi));
  return cb;
}

static void cb_submit_wait(VkCommandBuffer cb)
{
  CHECK(vkEndCommandBuffer(cb));
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1, .pCommandBuffers = &cb };
  CHECK(vkQueueSubmit(g_q, 1, &si, VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(g_q));
  vkFreeCommandBuffers(g_dev, g_pool, 1, &cb);
}

static void barrier(VkCommandBuffer cb)
{
  VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                     VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT };
  vkCmdPipelineBarrier(cb,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    0, 1, &mb, 0, NULL, 0, NULL);
}

typedef union { int32_t i; float f; } PcW;

static void dispatch_k(VkCommandBuffer cb, int kid, VkDescriptorSet ds,
                       const PcW *pc, int npc,
                       uint32_t gx, uint32_t gy, uint32_t gz, const char *label)
{
  Kern *k = &g_k[kid];
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pl, 0, 1, &ds, 0, NULL);
  if(npc) vkCmdPushConstants(cb, k->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             (uint32_t)(npc * 4), pc);
  if(g_ts_n < 255)
  {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_qpool, g_ts_n);
    g_ts_label[g_ts_n] = label;
  }
  vkCmdDispatch(cb, gx, gy, gz);
  if(g_ts_n < 255)
  {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qpool, g_ts_n + 1);
    g_ts_n += 2;
  }
  barrier(cb);
}

#define AUP(n, a) ((((n) + (a) - 1) / (a)))

/* [TDR-safe] per-submission row banding (same design as galosh_vk.c:
 * preemption unit = the SUBMISSION).  `rate` is a per-kernel learned
 * ms/WG-row state (pass12 and loess have very different costs — separate
 * states, unlike the RAW host's single global). */
static VkCommandBuffer band_piece(int kid, VkDescriptorSet ds, const PcW *pc, int npc,
                       uint32_t gx, uint32_t y0, uint32_t rows,
                       int ts_open, int ts_close, const char *label)
{
  Kern *k = &g_k[kid];
  VkCommandBuffer cb = cb_begin();
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pl, 0, 1, &ds, 0, NULL);
  if(npc) vkCmdPushConstants(cb, k->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             (uint32_t)(npc * 4), pc);
  if(ts_open && g_ts_n < 255)
  {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_qpool, g_ts_n);
    g_ts_label[g_ts_n] = label;
  }
  vkCmdDispatchBase(cb, 0, y0, 0, gx, rows, 1);
  if(ts_close && g_ts_n < 255)
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qpool, g_ts_n + 1);
  barrier(cb);
  CHECK(vkEndCommandBuffer(cb));
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1, .pCommandBuffers = &cb };
  CHECK(vkQueueSubmit(g_q, 1, &si, VK_NULL_HANDLE));
  return cb;
}

static void dispatch_k_banded(int kid, VkDescriptorSet ds,
                              const PcW *pc, int npc,
                              uint32_t gx, uint32_t gy, const char *label,
                              float *rate /* per-kernel ms/WG-row state */)
{
  const uint32_t probe_rows = (gy > 2) ? 2u : gy;
  VkCommandBuffer cbs[64];
  uint32_t ncb = 0;
  const clock_t t_all = clock();
  double per_row_ms;

  if(*rate > 0.0f)
  {
    per_row_ms = (double)*rate;
    const double est_ms = per_row_ms * (double)gy;
    uint32_t pieces = (est_ms > 800.0) ? (uint32_t)(est_ms / 800.0) + 1 : 1;
    if(pieces > gy) pieces = gy;
    if(pieces > 64) pieces = 64;
    for(uint32_t p = 0; p < pieces; p++)
    {
      const uint32_t a = gy * p / pieces, b = gy * (p + 1) / pieces;
      if(b > a) cbs[ncb++] = band_piece(kid, ds, pc, npc, gx, a, b - a,
                                        p == 0, p == pieces - 1, label);
    }
    CHECK(vkQueueWaitIdle(g_q));
  }
  else
  {
    cbs[ncb++] = band_piece(kid, ds, pc, npc, gx, 0, probe_rows, 1,
                            probe_rows >= gy, label);
    CHECK(vkQueueWaitIdle(g_q));
    const double probe_ms = 1000.0 * (double)(clock() - t_all) / CLOCKS_PER_SEC;
    if(probe_rows < gy)
    {
      const uint32_t rest_y0 = probe_rows, rest = gy - probe_rows;
      per_row_ms = probe_ms / (double)probe_rows;
      const double est_rest_ms = per_row_ms * (double)rest;
      uint32_t pieces = (est_rest_ms > 800.0) ? (uint32_t)(est_rest_ms / 800.0) + 1 : 1;
      if(pieces > rest) pieces = rest;
      if(pieces > 63) pieces = 63;
      for(uint32_t p = 0; p < pieces; p++)
      {
        const uint32_t a = rest_y0 + rest * p / pieces;
        const uint32_t b = rest_y0 + rest * (p + 1) / pieces;
        if(b > a) cbs[ncb++] = band_piece(kid, ds, pc, npc, gx, a, b - a,
                                          0, p == pieces - 1, label);
      }
      CHECK(vkQueueWaitIdle(g_q));
    }
  }
  vkFreeCommandBuffers(g_dev, g_pool, ncb, cbs);
  const double total_ms = 1000.0 * (double)(clock() - t_all) / CLOCKS_PER_SEC;
  *rate = (float)(total_ms / (double)gy);
  if(g_ts_n < 255) g_ts_n += 2;
}
static float g_rate_p12 = 0.0f, g_rate_loess = 0.0f;

/* [video] --noise=fit|hold|every:N + --noise-state=FILE — same semantics
 * as the RAW Vulkan host (galosh_vk.c B5).  YUV holds THREE scalars
 * (alpha, sigma_sq, sigma_gat) instead of RAW's two: sigma_gat is the
 * expensive one here (the 2nd serial-quickselect lap_mad, ~21 ms at 4K)
 * and it is a function of the held noise model — holding it is the whole
 * point of the port.  A held frame skips BOTH lap_mads, synth_alpha and
 * (with the .lut sidecar) the LUT build.  State file per core-run domain:
 * "<state><tag>" where tag = "" (444) / "_luma" / "_chroma" (420 — the
 * two sub-runs see different image domains, so separate states).
 * (日) YUV は σ_gat も保持（ここが 4K 21ms の本丸）。420 はルマ/クロマで
 * 状態ファイルを分離。--noise=ema は v1 未対応（中間 readback が要る）。 */
static const char *g_noise_mode = "fit";
static const char *g_noise_state = NULL;

/* ================= staging I/O ================= */
static Buf g_stg; static void *g_stg_map;

/* [embed] grow-only staging (re)allocation — replaces the fixed-size
 * creation the CLI used to do in main(); lets one device serve frames of
 * any size (frameserver plugin) and multiple planar sub-runs. */
static void ensure_staging(VkDeviceSize need)
{
  if(need < 1024) need = 1024;
  if(g_stg.buf && g_stg.size >= need) return;
  if(g_stg.buf)
  {
    vkUnmapMemory(g_dev, g_stg.mem);
    vkDestroyBuffer(g_dev, g_stg.buf, NULL);
    vkFreeMemory(g_dev, g_stg.mem, NULL);
  }
  g_stg = mkbuf(need,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  CHECK(vkMapMemory(g_dev, g_stg.mem, 0, VK_WHOLE_SIZE, 0, &g_stg_map));
}

/* [embed] in-memory noise state — the frameserver plugin holds the blind
 * fit per filter instance instead of the CLI's --noise-state files.
 * mem == NULL keeps the file-based CLI behavior exactly. */
typedef struct { int valid; float alpha, sigma_sq, sigma_gat; } galosh_vk_memstate;

/* [embed] per-frame log suppression for host processes (video spam). */
static int g_vk_quiet = 0;

static void upload(Buf *dst, const void *src, VkDeviceSize sz, VkDeviceSize dst_off)
{
  memcpy(g_stg_map, src, sz);
  VkCommandBuffer cb = cb_begin();
  VkBufferCopy c = { .srcOffset = 0, .dstOffset = dst_off, .size = sz };
  vkCmdCopyBuffer(cb, g_stg.buf, dst->buf, 1, &c);
  cb_submit_wait(cb);
}

static void download(Buf *src, void *dst, VkDeviceSize sz, VkDeviceSize src_off)
{
  VkCommandBuffer cb = cb_begin();
  VkBufferCopy c = { .srcOffset = src_off, .dstOffset = 0, .size = sz };
  vkCmdCopyBuffer(cb, src->buf, g_stg.buf, 1, &c);
  cb_submit_wait(cb);
  memcpy(dst, g_stg_map, sz);
}

/* ================= device init (once per process) =================
 * EN: Factored out of main() so embedders (the GALOSH-frameserver
 *     plugin) can bring the device up without the CLI: instance +
 *     deterministic device selection (GALOSH_VK_DEVICE honored) +
 *     subgroup probe + pipelines.  Idempotent; staging is allocated
 *     lazily by ensure_staging().  Returns 0 on success.
 * JP: main() から切り出し（プラグイン組み込み用）。冪等。
 * ================================================================ */
static int g_vk_inited = 0;
static int galosh_yuv_vk_init_device(void)
{
  if(g_vk_inited) return 0;
  /* --- instance / device (deterministic order + name match; same policy
   * as galosh_vk.c — see its comment on unstable enumeration) --- */
  VkApplicationInfo aiInfo = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "galosh-yuv-vk", .apiVersion = VK_API_VERSION_1_2 };
  VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &aiInfo };
  CHECK(vkCreateInstance(&ici, NULL, &g_inst));
  uint32_t nd = 0; vkEnumeratePhysicalDevices(g_inst, &nd, NULL);
  VkPhysicalDevice pds[8]; if(nd > 8) nd = 8;
  vkEnumeratePhysicalDevices(g_inst, &nd, pds);
  VkPhysicalDeviceProperties pp[8];
  uint32_t order[8];
  for(uint32_t i = 0; i < nd; i++)
  { vkGetPhysicalDeviceProperties(pds[i], &pp[i]); order[i] = i; }
  for(uint32_t a = 0; a + 1 < nd; a++)
    for(uint32_t b = a + 1; b < nd; b++)
    {
      #define TYPE_RANK(t) ((t) == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 0 : \
                            (t) == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 2)
      const int ra = TYPE_RANK(pp[order[a]].deviceType);
      const int rb = TYPE_RANK(pp[order[b]].deviceType);
      if(rb < ra || (rb == ra &&
         strcmp(pp[order[b]].deviceName, pp[order[a]].deviceName) < 0))
      { uint32_t t = order[a]; order[a] = order[b]; order[b] = t; }
    }
  uint32_t di = order[0];
  { const char *e = getenv("GALOSH_VK_DEVICE");
    if(e && e[0])
    {
      if(e[0] >= '0' && e[0] <= '9')
      { uint32_t w = (uint32_t)atoi(e); if(w < nd) di = order[w]; }
      else
      {
        for(uint32_t i = 0; i < nd; i++)
        {
          const char *hay = pp[i].deviceName;
          for(const char *h = hay; *h; h++)
          {
            const char *n = e, *q = h;
            while(*n && *q && ((*n | 32) == (*q | 32))) { n++; q++; }
            if(!*n) { di = i; goto dev_found; }
          }
        }
        dev_found: ;
      }
    }
  }
  g_pd = pds[di];
  vkGetPhysicalDeviceProperties(g_pd, &g_props);
  fprintf(stderr, "[yuv_vk] device[%u] = %s\n", di, g_props.deviceName);

  /* [B7g] subgroup-size-control probe (pass12_sg eligibility) */
  {
    uint32_t nx = 0;
    vkEnumerateDeviceExtensionProperties(g_pd, NULL, &nx, NULL);
    VkExtensionProperties *xp = malloc(nx * sizeof *xp);
    vkEnumerateDeviceExtensionProperties(g_pd, NULL, &nx, xp);
    int has = 0;
    for(uint32_t i = 0; i < nx; i++)
      if(strcmp(xp[i].extensionName, "VK_EXT_subgroup_size_control") == 0) has = 1;
    free(xp);
    if(has)
    {
      VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT };
      VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &sf };
      vkGetPhysicalDeviceFeatures2(g_pd, &f2);
      VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT };
      VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sp };
      vkGetPhysicalDeviceProperties2(g_pd, &p2);
      g_sg_ok = (sf.subgroupSizeControl && sf.computeFullSubgroups &&
                 sp.minSubgroupSize <= 32 && 32 <= sp.maxSubgroupSize &&
                 (sp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) ? 1 : 0;
    }
  }

  uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &qn, NULL);
  VkQueueFamilyProperties qf[16]; if(qn > 16) qn = 16;
  vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &qn, qf);
  g_qi = 0;
  for(uint32_t i = 0; i < qn; i++)
    if(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_qi = i; break; }
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_qi, .queueCount = 1, .pQueuePriorities = &prio };
  VkPhysicalDevice16BitStorageFeatures f16stor = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    .storageBuffer16BitAccess = VK_TRUE };
  VkPhysicalDeviceShaderFloat16Int8Features f16arith = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    .pNext = &f16stor, .shaderFloat16 = VK_TRUE };
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgcf = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
    .subgroupSizeControl = VK_TRUE, .computeFullSubgroups = VK_TRUE };
  if(g_sg_ok) f16stor.pNext = &sgcf;
  const char *dev_exts[1] = { "VK_EXT_subgroup_size_control" };
  VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = &f16arith,
    .enabledExtensionCount = g_sg_ok ? 1u : 0u,
    .ppEnabledExtensionNames = dev_exts,
    .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .pEnabledFeatures = NULL };
  CHECK(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
  vkGetDeviceQueue(g_dev, g_qi, 0, &g_q);
  VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qi };
  CHECK(vkCreateCommandPool(g_dev, &cpci, NULL, &g_pool));
  VkDescriptorPoolSize dps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               .descriptorCount = 256 };
  VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 64, .poolSizeCount = 1, .pPoolSizes = &dps };
  CHECK(vkCreateDescriptorPool(g_dev, &dpi, NULL, &g_dpool));
  VkQueryPoolCreateInfo qpci = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 256 };
  CHECK(vkCreateQueryPool(g_dev, &qpci, NULL, &g_qpool));

  for(int i = 0; i < K_COUNT; i++)
  {
    if(g_k[i].sg32 && !g_sg_ok) continue;
    make_kernel(&g_k[i]);
  }

  g_vk_inited = 1;
  return 0;
}

/* ================= core: one YUV frame on device ================= */
/* srgb: caller-owned 3ch interleaved f32 in/out (gamma sRGB domain —
 * exactly the OpenCL run_yuv_gat_gpu_buf contract).  luma_only skips the
 * chroma stage per YUV_BLUEPRINT.md §7.2 (the 420 gray pass / --pix=400). */
static int run_core(float *srgb, const int W, const int H,
                    const float strength_y, const float strength_c,
                    const int luma_only, const char *state_tag,
                    galosh_vk_memstate *mem)
{
  const size_t npix = (size_t)W * H;
  const size_t fb = npix * 4, hb = npix * 2;
  const int npix_i = (int)npix;
  const double t0 = (double)clock() / CLOCKS_PER_SEC;
  g_ts_n = 0;
  ensure_staging(3 * fb);

  /* ---- [video] noise-state load + hold decision ---- */
  char spath[1400] = { 0 }, lpath[1420] = { 0 };
  float st_a = 0.0f, st_s = 0.0f, st_g = 0.0f;
  int st_n = -1;
  if(mem)
  {
    /* [embed] in-memory state (frameserver): held values live in the
     * caller's instance struct; the noise-mode decision is the caller's
     * (mem->valid == use held model; else fit and store back). */
    if(mem->valid) { st_a = mem->alpha; st_s = mem->sigma_sq; st_g = mem->sigma_gat; st_n = 0; }
  }
  else if(g_noise_state)
  {
    snprintf(spath, sizeof(spath), "%s%s", g_noise_state, state_tag);
    snprintf(lpath, sizeof(lpath), "%s.lut", spath);
    FILE *sf = fopen(spath, "r");
    if(sf)
    {
      if(fscanf(sf, "%f %f %f %d", &st_a, &st_s, &st_g, &st_n) != 4) st_n = -1;
      float r = 0.0f;
      if(fscanf(sf, " rate12 %f", &r) == 1 && r > 0.0f) g_rate_p12 = r;
      r = 0.0f;
      if(fscanf(sf, " rateloess %f", &r) == 1 && r > 0.0f) g_rate_loess = r;
      fclose(sf);
    }
  }
  const int have_state = (st_n >= 0 && st_a > 0.0f && st_g > 0.0f);
  int hold_frame = 0;
  if(mem)
    hold_frame = have_state;
  else if(strcmp(g_noise_mode, "hold") == 0)
  {
    if(!have_state)
    { fprintf(stderr, "[yuv_vk] --noise=hold needs a valid --noise-state\n"); return 1; }
    hold_frame = 1;
  }
  else if(strncmp(g_noise_mode, "every:", 6) == 0)
  {
    const int N = atoi(g_noise_mode + 6);
    hold_frame = (have_state && st_n + 1 < (N > 0 ? N : 1));
  }
  else if(strncmp(g_noise_mode, "ema", 3) == 0)
  { fprintf(stderr, "[yuv_vk] --noise=ema not supported (v1: fit|hold|every:N)\n"); return 1; }

  /* per-run buffers (sizes are frame-dependent; device/pipelines persist) */
  Buf srgb_b = DEVBUF(3 * fb);
  Buf y_lin  = DEVBUF(fb);
  Buf y_stab = DEVBUF(hb), y_den = DEVBUF(hb), y_snap = DEVBUF(hb);
  Buf cb_b = DEVBUF(hb), cr_b = DEVBUF(hb);
  Buf cb_den = DEVBUF(hb), cr_den = DEVBUF(hb);
  Buf ne_scr = DEVBUF((200000 + 64) * 4);
  Buf params = DEVBUF(PARAMS_SIZE * 4);
  Buf lut_d = DEVBUF(GAT_LUT_SIZE * 4), lut_x = DEVBUF(GAT_LUT_SIZE * 4);
  Buf lut_p = DEVBUF(8 * 4);

  upload(&srgb_b, srgb, 3 * fb, 0);
  float h_params[PARAMS_SIZE] = { 0 };
  if(hold_frame)   /* held model lands directly in the initial upload */
  {
    h_params[P_ALPHA]     = st_a;
    h_params[P_SIGMA_SQ]  = st_s;
    h_params[P_SIGMA_GAT] = st_g;
  }
  upload(&params, h_params, PARAMS_SIZE * 4, 0);

  /* [video] LUT sidecar: on hold frames load the 32 KB cache and skip the
   * device rebuild; a missing sidecar is fine (alpha/sigma_sq are already
   * in params, so the on-device build still works). */
  int lut_from_cache = 0;
  if(hold_frame && lpath[0])
  {
    FILE *cf = fopen(lpath, "rb");
    if(cf)
    {
      float *tmp = malloc((GAT_LUT_SIZE * 2 + 8) * 4);
      if(fread(tmp, 4, GAT_LUT_SIZE * 2 + 8, cf) == (size_t)(GAT_LUT_SIZE * 2 + 8))
        lut_from_cache = 1;
      fclose(cf);
      if(lut_from_cache)
      {
        upload(&lut_d, tmp,                    GAT_LUT_SIZE * 4, 0);
        upload(&lut_x, tmp + GAT_LUT_SIZE,     GAT_LUT_SIZE * 4, 0);
        upload(&lut_p, tmp + 2 * GAT_LUT_SIZE, 8 * 4, 0);
      }
      free(tmp);
    }
  }

#define SET(kid, ...) mkset(kid, (const Buf *[]){ __VA_ARGS__ }, g_k[kid].nbind)
  VkDescriptorSet s_dec   = SET(K_SRGB2YCC, &srgb_b, &y_lin, &cb_b, &cr_b);
  VkDescriptorSet s_mad   = SET(K_LAP_MAD, &y_lin, &params, &ne_scr);
  VkDescriptorSet s_mad16 = SET(K_LAP_MAD_H16, &y_stab, &params, &ne_scr);
  VkDescriptorSet s_syn   = SET(K_SYNTH, &params);
  VkDescriptorSet s_gat   = SET(K_GAT_FWD, &y_lin, &y_stab, &params);
  VkDescriptorSet s_norm  = SET(K_NORM, &y_stab, &params);
  VkDescriptorSet s_lut   = SET(K_LUT_BUILD, &params, &lut_d, &lut_x, &lut_p);
  VkDescriptorSet s_lutf  = SET(K_LUT_FIN, &lut_d, &lut_p);
  const char *sg_env = getenv("GALOSH_SG");
  const int use_sg = g_sg_ok && !(sg_env && sg_env[0] == '0');
  const int kid_p12 = use_sg ? K_PASS12_SG : K_PASS12;
  VkDescriptorSet s_p12   = SET(kid_p12, &y_stab, &y_den);
  VkDescriptorSet s_den   = SET(K_DENORM, &y_den, &params);
  VkDescriptorSet s_mak   = SET(K_MAKITALO, &y_den, &y_lin, &lut_d, &lut_x, &lut_p);
  VkDescriptorSet s_lo    = SET(K_LOESS, &y_snap, &cb_b, &cr_b, &cb_den, &cr_den);
  /* recompose: chroma = LOESS out, or the untouched noisy planes when
   * luma_only (420 gray pass — blueprint §7.2 rebind) */
  VkDescriptorSet s_rec   = luma_only
      ? SET(K_YCC2SRGB, &y_lin, &cb_b, &cr_b, &srgb_b)
      : SET(K_YCC2SRGB, &y_lin, &cb_den, &cr_den, &srgb_b);

  PcW pc[4];
#define PCI(k, v) pc[k].i = (v)
#define PCF(k, v) pc[k].f = (v)

  /* ---- SEG 1: decompose + noise est + GAT + norm + LUT (one submit,
   * ZERO mid-frame readback — blueprint §5.2 deviation) ---- */
  VkCommandBuffer cb = cb_begin();
  PCI(0, npix_i);
  dispatch_k(cb, K_SRGB2YCC, s_dec, pc, 1,
             (uint32_t)AUP(npix, 256), 1, 1, "YG1 srgb2ycc");
  if(!hold_frame)   /* [video] held frames skip BOTH quickselects + synth */
  {
    PCI(0, W); PCI(1, H); PCI(2, 3); PCI(3, P_SIGMA_Y);
    dispatch_k(cb, K_LAP_MAD, s_mad, pc, 4, 1, 1, 1, "YGQ2a lap_mad_lin");
    PCI(0, P_SIGMA_Y); PCI(1, P_ALPHA); PCI(2, P_SIGMA_SQ);
    dispatch_k(cb, K_SYNTH, s_syn, pc, 3, 1, 1, 1, "YGQ2b synth_alpha");
  }
  PCI(0, npix_i);
  dispatch_k(cb, K_GAT_FWD, s_gat, pc, 1,
             (uint32_t)AUP(npix, 256), 1, 1, "YG4a gat_fwd");
  if(!hold_frame)
  {
    PCI(0, W); PCI(1, H); PCI(2, 3); PCI(3, P_SIGMA_GAT);
    dispatch_k(cb, K_LAP_MAD_H16, s_mad16, pc, 4, 1, 1, 1, "YG4a' lap_mad_gat");
  }
  PCI(0, P_SIGMA_GAT); PCI(1, npix_i);
  dispatch_k(cb, K_NORM, s_norm, pc, 2,
             (uint32_t)AUP(npix, 256), 1, 1, "YG4a'' norm");
  if(!luma_only)
  {
    /* snapshot the normalized post-GAT NOISY Y_stab = LOESS guide
     * (guide-domain trap, blueprint §5.3 — BEFORE pass12 overwrites) */
    VkBufferCopy c = { .srcOffset = 0, .dstOffset = 0, .size = hb };
    vkCmdCopyBuffer(cb, y_stab.buf, y_snap.buf, 1, &c);
    barrier(cb);
  }
  if(!lut_from_cache)
  {
    dispatch_k(cb, K_LUT_BUILD, s_lut, NULL, 0, GAT_LUT_SIZE / 256, 1, 1, "YG4c build_lut");
    dispatch_k(cb, K_LUT_FIN, s_lutf, NULL, 0, 1, 1, 1, "YG4d lut_fin");
  }
  cb_submit_wait(cb);

  /* ---- pass12 (banded, TDR-safe) ---- */
  PCI(0, W); PCI(1, H); PCF(2, strength_y); PCI(3, 1 /* phase_stride */);
  dispatch_k_banded(kid_p12, s_p12, pc, 4,
             (uint32_t)AUP(W, O32_TILE), (uint32_t)AUP(H, O32_TILE),
             "YG4f pass12", &g_rate_p12);

  /* ---- SEG 2: denorm + inverse GAT (y_den in place; the OpenCL
   * copy-back to y_stab is skipped — documented deviation) ---- */
  cb = cb_begin();
  PCI(0, P_SIGMA_GAT); PCI(1, npix_i);
  dispatch_k(cb, K_DENORM, s_den, pc, 2,
             (uint32_t)AUP(npix, 256), 1, 1, "YG4g' denorm");
  PCI(0, npix_i);
  dispatch_k(cb, K_MAKITALO, s_mak, pc, 1,
             (uint32_t)AUP(npix, 256), 1, 1, "YG4h makitalo");
  cb_submit_wait(cb);

  /* ---- chroma LOESS (banded, heavy) ---- */
  if(!luma_only)
  {
    PCI(0, W); PCI(1, H); PCF(2, strength_c);
    dispatch_k_banded(K_LOESS, s_lo, pc, 3,
               (uint32_t)AUP(W, 16), (uint32_t)AUP(H, 16),
               "YG5 loess", &g_rate_loess);
  }

  /* ---- recompose + download ---- */
  cb = cb_begin();
  PCI(0, npix_i);
  dispatch_k(cb, K_YCC2SRGB, s_rec, pc, 1,
             (uint32_t)AUP(npix, 256), 1, 1, "YG9 ycc2srgb");
  cb_submit_wait(cb);
  download(&srgb_b, srgb, 3 * fb, 0);

  /* log the on-device blind fit (readback AFTER the frame — logging only) */
  download(&params, h_params, PARAMS_SIZE * 4, 0);
  if(!g_vk_quiet)
    fprintf(stderr, "[yuv_vk] %dx%d blind sigma=%.5f alpha=%.6g sigma_sq=%.6e "
                    "sigma_gat=%.4f s_y=%.2f s_c=%.2f%s%s\n",
            W, H, h_params[P_SIGMA_Y], h_params[P_ALPHA], h_params[P_SIGMA_SQ],
            h_params[P_SIGMA_GAT], strength_y, strength_c,
            luma_only ? " (luma-only)" : "",
            hold_frame ? " [noise HOLD]" : "");

  /* [embed] write the fitted model back to the caller's in-memory state */
  if(mem && !hold_frame)
  {
    mem->alpha = h_params[P_ALPHA];
    mem->sigma_sq = h_params[P_SIGMA_SQ];
    mem->sigma_gat = h_params[P_SIGMA_GAT];
    mem->valid = 1;
  }

  /* [video] persist state (+ 32 KB LUT sidecar on fit frames) */
  if(!mem && spath[0] && strcmp(g_noise_mode, "fit") != 0)
  {
    const int new_n = hold_frame ? st_n + 1 : 0;
    FILE *sf = fopen(spath, "w");
    if(sf)
    {
      fprintf(sf, "%.9g %.9g %.9g %d\n", h_params[P_ALPHA],
              h_params[P_SIGMA_SQ], h_params[P_SIGMA_GAT], new_n);
      if(g_rate_p12 > 0.0f)   fprintf(sf, "rate12 %.6g\n", g_rate_p12);
      if(g_rate_loess > 0.0f) fprintf(sf, "rateloess %.6g\n", g_rate_loess);
      fclose(sf);
    }
    if(!lut_from_cache && lpath[0])
    {
      float *tmp = malloc((GAT_LUT_SIZE * 2 + 8) * 4);
      download(&lut_d, tmp, GAT_LUT_SIZE * 4, 0);
      download(&lut_x, tmp + GAT_LUT_SIZE, GAT_LUT_SIZE * 4, 0);
      download(&lut_p, tmp + 2 * GAT_LUT_SIZE, 8 * 4, 0);
      FILE *cf = fopen(lpath, "wb");
      if(cf) { fwrite(tmp, 4, GAT_LUT_SIZE * 2 + 8, cf); fclose(cf); }
      free(tmp);
    }
  }

  if(g_verbose && g_ts_n)
  {
    uint64_t ts[256];
    CHECK(vkGetQueryPoolResults(g_dev, g_qpool, 0, g_ts_n, sizeof(uint64_t) * g_ts_n,
                                ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    double total = 0;
    fprintf(stderr, "[yuv_vk] ====== PER-KERNEL PROFILING ======\n");
    for(uint32_t i = 0; i + 1 < g_ts_n; i += 2)
    {
      double ms = (double)(ts[i + 1] - ts[i]) * g_props.limits.timestampPeriod * 1e-6;
      total += ms;
      fprintf(stderr, "[yuv_vk]   %-28s %8.3f ms\n", g_ts_label[i], ms);
    }
    fprintf(stderr, "[yuv_vk]   TOTAL (GPU time)  %.3f ms\n", total);
  }
  if(!g_vk_quiet)
    fprintf(stderr, "[yuv_vk] frame total %.1f ms\n",
            1000.0 * ((double)clock() / CLOCKS_PER_SEC - t0));

  freebuf(&srgb_b); freebuf(&y_lin);
  freebuf(&y_stab); freebuf(&y_den); freebuf(&y_snap);
  freebuf(&cb_b); freebuf(&cr_b); freebuf(&cb_den); freebuf(&cr_den);
  freebuf(&ne_scr); freebuf(&params);
  freebuf(&lut_d); freebuf(&lut_x); freebuf(&lut_p);
  CHECK(vkResetDescriptorPool(g_dev, g_dpool, 0));
  return 0;
}

#ifndef GALOSH_YUV_VK_NOMAIN
/* ================= GALOSH-420 planar driver =================
 * Identical composition to the CPU / OpenCL drivers (galosh_yuv420.h is
 * the single shared front-end; YUV_BLUEPRINT.md §7).  ONE device init,
 * core runs twice: full-res gray LUMA-ONLY pass + native-lattice chroma
 * pass.  Format-preserving planar I/O. */
static int run_420(const char *in_path, const char *out_path,
                   const int W, const int H,
                   const float sy, const float sc,
                   const galosh420_pix_t pix, const galosh420_siting_t siting,
                   const galosh420_eotf_t eotf, const galosh420_matrix_t mat,
                   const galosh420_range_t range, const int depth)
{
  if(pix == GALOSH420_PIX_420 && ((W | H) & 1))
  { fprintf(stderr, "[yuv_vk420] 4:2:0 requires even W/H\n"); return 1; }
  if(pix == GALOSH420_PIX_422 && (W & 1))
  { fprintf(stderr, "[yuv_vk420] 4:2:2 requires even W\n"); return 1; }

  const int cw = (pix == GALOSH420_PIX_400) ? 0
               : (pix == GALOSH420_PIX_444) ? W : W / 2;
  const int ch = (pix == GALOSH420_PIX_400) ? 0
               : (pix == GALOSH420_PIX_420) ? H / 2 : H;
  const size_t ysz = (size_t)W * H;
  const size_t csz = (size_t)cw * ch;
  const int wide = (depth > 8);
  const size_t code_bytes = (ysz + 2 * csz) * (wide ? 2 : 1);

  uint8_t *raw = (uint8_t *)malloc(code_bytes);
  float *Yp = (float *)malloc(ysz * 4);
  float *Cb = csz ? (float *)malloc(csz * 4) : NULL;
  float *Cr = csz ? (float *)malloc(csz * 4) : NULL;
  if(!raw || !Yp || (csz && (!Cb || !Cr))) { fprintf(stderr, "alloc failed\n"); return 1; }

  FILE *fi = fopen(in_path, "rb");
  if(!fi) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
  const size_t rd = fread(raw, 1, code_bytes, fi);
  fclose(fi);
  if(rd != code_bytes)
  { fprintf(stderr, "[yuv_vk420] short read (%zu of %zu)\n", rd, code_bytes); return 1; }

  const uint16_t *raw16 = (const uint16_t *)raw;
  for(size_t i = 0; i < ysz; i++)
    Yp[i] = galosh420_dequant_y(wide ? (float)raw16[i] : (float)raw[i], depth, range);
  for(size_t i = 0; i < csz; i++)
  {
    Cb[i] = galosh420_dequant_c(wide ? (float)raw16[ysz + i]
                                     : (float)raw[ysz + i], depth, range);
    Cr[i] = galosh420_dequant_c(wide ? (float)raw16[ysz + csz + i]
                                     : (float)raw[ysz + csz + i], depth, range);
  }

  /* ---- LUMA (full res, gray image, luma-only fast path) ---- */
  float *gray = (float *)malloc(ysz * 3 * 4);
  if(!gray) { fprintf(stderr, "alloc failed\n"); return 1; }
  for(size_t i = 0; i < ysz; i++)
  {
    const float v = galosh420_eotf_fwd_f(
        fminf(fmaxf(galosh420_eotf_inv_f(Yp[i], eotf), 0.0f), 1.0f),
        GALOSH420_EOTF_SRGB);
    gray[3 * i + 0] = v; gray[3 * i + 1] = v; gray[3 * i + 2] = v;
  }
  if(run_core(gray, W, H, sy, sc, /*luma_only=*/1, "_luma", NULL)) return 1;
  float *Ydeng = (float *)malloc(ysz * 4);
  if(!Ydeng) { fprintf(stderr, "alloc failed\n"); return 1; }
  for(size_t i = 0; i < ysz; i++)
  {
    const float lin = galosh420_eotf_inv_f(gray[3 * i + 1], GALOSH420_EOTF_SRGB);
    Ydeng[i] = galosh420_eotf_fwd_f(fminf(fmaxf(lin, 0.0f), 1.0f), eotf);
  }
  free(gray);

  /* ---- CHROMA (native lattice) ---- */
  float *CbD = NULL, *CrD = NULL;
  if(pix != GALOSH420_PIX_400)
  {
    const int pw = (pix == GALOSH420_PIX_420) ? cw : W;
    const int ph = (pix == GALOSH420_PIX_420) ? ch : H;
    const size_t psz = (size_t)pw * ph;
    float *Yg  = (float *)malloc(psz * 4);
    float *Cbp = (float *)malloc(psz * 4);
    float *Crp = (float *)malloc(psz * 4);
    float *rgb = (float *)malloc(psz * 3 * 4);
    if(!Yg || !Cbp || !Crp || !rgb) { fprintf(stderr, "alloc failed\n"); return 1; }

    if(pix == GALOSH420_PIX_420)
    {
      galosh420_down_luma(Yp, W, H, Yg, siting);
      memcpy(Cbp, Cb, csz * 4);
      memcpy(Crp, Cr, csz * 4);
    }
    else if(pix == GALOSH420_PIX_422)
    {
      memcpy(Yg, Yp, ysz * 4);
      galosh420_up422_h(Cb, cw, H, Cbp);
      galosh420_up422_h(Cr, cw, H, Crp);
    }
    else
    {
      memcpy(Yg, Yp, ysz * 4);
      memcpy(Cbp, Cb, csz * 4);
      memcpy(Crp, Cr, csz * 4);
    }
    for(size_t i = 0; i < psz; i++)
    {
      float R, G, B;
      galosh420_ncl_inv(Yg[i], Cbp[i], Crp[i], mat, &R, &G, &B);
      rgb[3 * i + 0] = galosh420_eotf_fwd_f(galosh420_eotf_inv_f(R, eotf), GALOSH420_EOTF_SRGB);
      rgb[3 * i + 1] = galosh420_eotf_fwd_f(galosh420_eotf_inv_f(G, eotf), GALOSH420_EOTF_SRGB);
      rgb[3 * i + 2] = galosh420_eotf_fwd_f(galosh420_eotf_inv_f(B, eotf), GALOSH420_EOTF_SRGB);
    }
    if(run_core(rgb, pw, ph, sy, sc, /*luma_only=*/0, "_chroma", NULL)) return 1;
    for(size_t i = 0; i < psz; i++)
    {
      const float Rp = galosh420_eotf_fwd_f(
          galosh420_eotf_inv_f(rgb[3 * i + 0], GALOSH420_EOTF_SRGB), eotf);
      const float Gp = galosh420_eotf_fwd_f(
          galosh420_eotf_inv_f(rgb[3 * i + 1], GALOSH420_EOTF_SRGB), eotf);
      const float Bp = galosh420_eotf_fwd_f(
          galosh420_eotf_inv_f(rgb[3 * i + 2], GALOSH420_EOTF_SRGB), eotf);
      float yy;
      galosh420_ncl_fwd(Rp, Gp, Bp, mat, &yy, &Cbp[i], &Crp[i]);
    }
    CbD = (float *)malloc(csz * 4);
    CrD = (float *)malloc(csz * 4);
    if(!CbD || !CrD) { fprintf(stderr, "alloc failed\n"); return 1; }
    if(pix == GALOSH420_PIX_422)
    {
      galosh420_down422_h(Cbp, W, H, CbD);
      galosh420_down422_h(Crp, W, H, CrD);
    }
    else
    {
      memcpy(CbD, Cbp, csz * 4);
      memcpy(CrD, Crp, csz * 4);
    }
    free(Yg); free(Cbp); free(Crp); free(rgb);
  }

  /* ---- requantise + write (format-preserving) ---- */
  for(size_t i = 0; i < ysz; i++)
  {
    const int c = galosh420_requant_y(Ydeng[i], depth, range);
    if(wide) ((uint16_t *)raw)[i] = (uint16_t)c;
    else     raw[i] = (uint8_t)c;
  }
  for(size_t i = 0; i < csz; i++)
  {
    const int cb2 = galosh420_requant_c(CbD[i], depth, range);
    const int cr2 = galosh420_requant_c(CrD[i], depth, range);
    if(wide)
    {
      ((uint16_t *)raw)[ysz + i]       = (uint16_t)cb2;
      ((uint16_t *)raw)[ysz + csz + i] = (uint16_t)cr2;
    }
    else
    {
      raw[ysz + i]       = (uint8_t)cb2;
      raw[ysz + csz + i] = (uint8_t)cr2;
    }
  }
  FILE *fo = fopen(out_path, "wb");
  if(!fo) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
  const size_t wr = fwrite(raw, 1, code_bytes, fo);
  fclose(fo);
  if(wr != code_bytes) { fprintf(stderr, "short write\n"); return 1; }
  free(raw); free(Yp); free(Cb); free(Cr); free(Ydeng); free(CbD); free(CrD);
  return 0;
}

/* ================= main ================= */
int main(int argc, char **argv)
{
  {
    strncpy(g_exe_dir, argv[0], sizeof(g_exe_dir) - 1);
    char *s1 = strrchr(g_exe_dir, '\\'), *s2 = strrchr(g_exe_dir, '/');
    char *cut = s1 > s2 ? s1 : s2;
    if(cut) *cut = 0; else strcpy(g_exe_dir, ".");
  }
  g_verbose = getenv("GALOSH_VERBOSE") != NULL;

  galosh420_pix_t    pix    = GALOSH420_PIX_444;
  int                planar = 0;
  galosh420_siting_t siting = GALOSH420_SITING_CENTER;
  galosh420_eotf_t   eotf   = GALOSH420_EOTF_SRGB;
  galosh420_matrix_t mat    = GALOSH420_MAT_BT709;
  galosh420_range_t  range  = GALOSH420_RANGE_FULL;
  int                depth  = 8;

  char *pos[16]; int np = 0;
  for(int i = 1; i < argc && np < 16; i++)
  {
    const char *a = argv[i];
    if(strcmp(a, "--selftest-phase") == 0) return galosh420_phase_selftest();
    else if(strncmp(a, "--noise=", 8) == 0)        g_noise_mode  = a + 8;
    else if(strncmp(a, "--noise-state=", 14) == 0) g_noise_state = a + 14;
    else if(strncmp(a, "--pix=", 6) == 0)
    {
      const char *v = a + 6;
      if(!strcmp(v, "420"))      pix = GALOSH420_PIX_420;
      else if(!strcmp(v, "422")) pix = GALOSH420_PIX_422;
      else if(!strcmp(v, "400")) pix = GALOSH420_PIX_400;
      else if(!strcmp(v, "444")) pix = GALOSH420_PIX_444;
      else { fprintf(stderr, "bad --pix=%s\n", v); return 1; }
      planar = 1;
    }
    else if(strncmp(a, "--siting=", 9) == 0)
    { if(galosh420_parse_siting(a + 9, &siting)) { fprintf(stderr, "bad --siting\n"); return 1; } }
    else if(strncmp(a, "--eotf=", 7) == 0)
    { if(galosh420_parse_eotf(a + 7, &eotf)) { fprintf(stderr, "bad --eotf\n"); return 1; } }
    else if(strncmp(a, "--matrix=", 9) == 0)
    { if(galosh420_parse_matrix(a + 9, &mat)) { fprintf(stderr, "bad --matrix\n"); return 1; } }
    else if(strncmp(a, "--range=", 8) == 0)
    { if(galosh420_parse_range(a + 8, &range)) { fprintf(stderr, "bad --range\n"); return 1; } }
    else if(strncmp(a, "--depth=", 8) == 0)
    {
      depth = atoi(a + 8);
      if(depth < 8 || depth > 16) { fprintf(stderr, "bad --depth\n"); return 1; }
    }
    else if(strncmp(a, "--", 2) != 0) pos[np++] = (char *)a;
  }
  if(np < 4)
  {
    fprintf(stderr,
      "Usage: galosh_yuv_vk in.bin out.bin W H [s_y] [s_c]\n"
      "  [--pix=420|422|444|400] [--depth=8..16] [--range=full|limited]\n"
      "  [--matrix=bt601|bt709|bt2020|custom:Kr,Kb]\n"
      "  [--eotf=srgb|g22|g24|bt709|hlg|pq|linear]\n"
      "  [--siting=center|left|topleft] [--selftest-phase]\n");
    return 1;
  }
  const char *in_path = pos[0], *out_path = pos[1];
  const int W = atoi(pos[2]), H = atoi(pos[3]);
  const float sy = (np > 4) ? (float)atof(pos[4]) : 1.0f;
  const float sc = (np > 5) ? (float)atof(pos[5]) : 1.0f;
  if(W <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 1; }

  if(galosh_yuv_vk_init_device()) return 1;

  if(planar)
    return run_420(in_path, out_path, W, H, sy, sc,
                   pix, siting, eotf, mat, range, depth);

  /* legacy 444 float mode (CLI-compatible with galosh_yuv_gpu.exe) */
  const size_t npix = (size_t)W * H;
  float *img = (float *)malloc(npix * 3 * 4);
  if(!img) { fprintf(stderr, "alloc failed\n"); return 1; }
  { FILE *f = fopen(in_path, "rb");
    if(!f) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
    if(fread(img, 4, npix * 3, f) != npix * 3)
    { fprintf(stderr, "short input\n"); return 1; }
    fclose(f); }
  if(run_core(img, W, H, sy, sc, 0, "", NULL)) return 1;
  { FILE *f = fopen(out_path, "wb");
    if(!f) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    fwrite(img, 4, npix * 3, f); fclose(f); }
  free(img);
  fprintf(stderr, "[yuv_vk] done: %s (%dx%d)\n", out_path, W, H);
  return 0;
}
#endif /* GALOSH_YUV_VK_NOMAIN */
