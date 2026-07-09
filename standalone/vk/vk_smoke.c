/* vk_smoke.c — [V2.0 Phase B1] Vulkan compute skeleton + plumbing smoke test.
 *
 * EN: Exercises the exact host-side structure the GALOSH Vulkan port will
 *     use: device selection (GALOSH_VK_DEVICE env, default = first discrete
 *     GPU), one compute queue, host-visible staging + device-local storage
 *     buffers, SPIR-V pipeline from file, descriptor set, PRE-RECORDED
 *     command buffer (the CUDA-Graphs-equivalent: record once, submit per
 *     frame), timestamp queries (profiler from day one), dispatch, readback,
 *     numeric verification.
 * JP: GALOSH Vulkan 移植のホスト構造をそのまま先取りした配管スモーク。
 *     事前記録コマンドバッファ（Graphs 相当）＋ timestamp を初日から。
 *
 * Build:  gcc -O2 -std=c11 vk_smoke.c -o vk_smoke.exe -lvulkan-1
 * Run:    ./vk_smoke.exe        (GALOSH_VK_DEVICE=n to pick a device)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult _r = (x); if(_r != VK_SUCCESS) { \
  fprintf(stderr, "VK error %d at %s:%d\n", _r, __FILE__, __LINE__); exit(1); } } while(0)

static VkShaderModule load_spv(VkDevice dev, const char *path)
{
  FILE *f = fopen(path, "rb");
  if(!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  uint32_t *code = malloc(sz);
  fread(code, 1, sz, f); fclose(f);
  VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = (size_t)sz, .pCode = code };
  VkShaderModule m;
  CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
  free(code);
  return m;
}

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want)
{
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for(uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  fprintf(stderr, "no memory type (want 0x%x)\n", want); exit(1);
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; } VkBuf;

static VkBuf make_buf(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize sz,
                      VkBufferUsageFlags usage, VkMemoryPropertyFlags props)
{
  VkBuf b;
  VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = sz, .usage = usage,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
  CHECK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, b.buf, &mr);
  VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = mr.size,
                               .memoryTypeIndex = find_mem(pd, mr.memoryTypeBits, props) };
  CHECK(vkAllocateMemory(dev, &mai, NULL, &b.mem));
  CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
  return b;
}

int main(void)
{
  const uint32_t N = 1u << 20;              /* 1M floats */
  const VkDeviceSize SZ = N * sizeof(float);

  /* --- instance + device selection --- */
  VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .pApplicationName = "galosh-vk-smoke",
                           .apiVersion = VK_API_VERSION_1_2 };
  VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &ai };
  VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));

  uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, NULL);
  VkPhysicalDevice pds[8]; if(nd > 8) nd = 8;
  vkEnumeratePhysicalDevices(inst, &nd, pds);
  int want = -1;
  const char *env = getenv("GALOSH_VK_DEVICE");
  if(env) want = atoi(env);
  uint32_t di = 0;
  for(uint32_t i = 0; i < nd; i++)
  {
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
    if(want >= 0 ? (int)i == want
                 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    { di = i; if(want >= 0 || i == 0) break; if(want < 0) break; }
  }
  VkPhysicalDevice pd = pds[di];
  VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
  printf("[vk] device[%u] = %s\n", di, props.deviceName);

  uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
  VkQueueFamilyProperties qf[16]; if(qn > 16) qn = 16;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf);
  uint32_t qi = 0;
  for(uint32_t i = 0; i < qn; i++)
    if(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = qi, .queueCount = 1,
                                  .pQueuePriorities = &prio };
  VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
  VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);

  /* --- buffers: host staging (up+down) + device-local in/out --- */
  VkBuf stg = make_buf(dev, pd, SZ,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkBuf din = make_buf(dev, pd, SZ,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkBuf dout = make_buf(dev, pd, SZ,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  /* --- descriptor set layout / pipeline --- */
  VkDescriptorSetLayoutBinding binds[2] = {
    { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT } };
  VkDescriptorSetLayoutCreateInfo dsli = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 2, .pBindings = binds };
  VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));

  VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                              .offset = 0, .size = 4 };
  VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1, .pSetLayouts = &dsl,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
  VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

  VkShaderModule sm = load_spv(dev, "shaders/smoke.spv");
  VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
    .layout = pl };
  VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

  VkDescriptorPoolSize dps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 };
  VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
  VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
  VkDescriptorSetAllocateInfo dsa = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = dp, .descriptorSetCount = 1,
                                      .pSetLayouts = &dsl };
  VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dsa, &ds));
  VkDescriptorBufferInfo bi0 = { .buffer = din.buf,  .range = VK_WHOLE_SIZE };
  VkDescriptorBufferInfo bi1 = { .buffer = dout.buf, .range = VK_WHOLE_SIZE };
  VkWriteDescriptorSet wr[2] = {
    { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bi0 },
    { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bi1 } };
  vkUpdateDescriptorSets(dev, 2, wr, 0, NULL);

  /* --- timestamp query pool --- */
  VkQueryPoolCreateInfo qpi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 };
  VkQueryPool qp; CHECK(vkCreateQueryPool(dev, &qpi, NULL, &qp));

  /* --- command buffer: RECORD ONCE (Graphs-equivalent), submit many --- */
  VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .queueFamilyIndex = qi };
  VkCommandPool cp; CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
  VkCommandBufferAllocateInfo cba = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                      .commandPool = cp,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      .commandBufferCount = 1 };
  VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev, &cba, &cb));

  VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  CHECK(vkBeginCommandBuffer(cb, &cbi));
  vkCmdResetQueryPool(cb, qp, 0, 2);
  VkBufferCopy cpy = { .size = SZ };
  vkCmdCopyBuffer(cb, stg.buf, din.buf, 1, &cpy);
  VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &N);
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
  vkCmdDispatch(cb, (N + 255) / 256, 1, 1);
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
  VkMemoryBarrier mb2 = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb2, 0, NULL, 0, NULL);
  vkCmdCopyBuffer(cb, dout.buf, stg.buf, 1, &cpy);
  CHECK(vkEndCommandBuffer(cb));

  /* --- fill staging, submit, verify --- */
  float *map;
  CHECK(vkMapMemory(dev, stg.mem, 0, SZ, 0, (void **)&map));
  for(uint32_t i = 0; i < N; i++) map[i] = (float)i * 0.001f;

  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1, .pCommandBuffers = &cb };
  CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(q));

  uint64_t ts[2];
  CHECK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  double ms = (double)(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6;

  int bad = 0;
  for(uint32_t i = 0; i < N; i++)
  {
    const float want_v = 2.0f * ((float)i * 0.001f) + 1.0f;
    if(map[i] != want_v) { if(bad < 3) fprintf(stderr, "  [%u] %g != %g\n", i, map[i], want_v); bad++; }
  }
  vkUnmapMemory(dev, stg.mem);
  printf("[vk] dispatch %.3f ms (1M floats)  verify: %s (%d bad)\n",
         ms, bad ? "FAIL" : "PASS", bad);
  return bad ? 1 : 0;
}
