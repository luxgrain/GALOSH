/* vk_probe.c — [V2.0 Phase B0] Vulkan capability probe for the GALOSH port.
 *
 * EN: Enumerates Vulkan devices and prints exactly the capabilities the
 *     GALOSH Vulkan port depends on: compute queue, timestamp support
 *     (profiler from day one), shaderFloat16 + 16-bit storage (the FP16
 *     contract, docs/dataflow_spec.md §4), subgroup size/ops (warp-level
 *     reductions), and shared-memory budget (LDS tiling contracts).
 * JP: GALOSH Vulkan 移植が依存する能力だけを列挙・表示する B0 プローブ。
 *
 * Build:  gcc -O2 -std=c11 vk_probe.c -o vk_probe.exe -lvulkan-1
 */
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

int main(void)
{
  VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .pApplicationName = "galosh-vk-probe",
                           .apiVersion = VK_API_VERSION_1_2 };
  VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &ai };
  VkInstance inst;
  if(vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
  { fprintf(stderr, "vkCreateInstance failed\n"); return 1; }

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, NULL);
  VkPhysicalDevice devs[8];
  if(n > 8) n = 8;
  vkEnumeratePhysicalDevices(inst, &n, devs);
  printf("devices: %u\n", n);

  for(uint32_t i = 0; i < n; i++)
  {
    VkPhysicalDeviceSubgroupProperties sg =
      { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceProperties2 p2 =
      { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sg };
    vkGetPhysicalDeviceProperties2(devs[i], &p2);
    VkPhysicalDeviceProperties *p = &p2.properties;

    VkPhysicalDevice16BitStorageFeatures s16 =
      { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
    VkPhysicalDeviceShaderFloat16Int8Features f16 =
      { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .pNext = &s16 };
    VkPhysicalDeviceFeatures2 f2 =
      { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f16 };
    vkGetPhysicalDeviceFeatures2(devs[i], &f2);

    printf("[%u] %s\n", i, p->deviceName);
    printf("    api %u.%u.%u  type %d\n",
           VK_API_VERSION_MAJOR(p->apiVersion),
           VK_API_VERSION_MINOR(p->apiVersion),
           VK_API_VERSION_PATCH(p->apiVersion), p->deviceType);
    printf("    shaderFloat16=%d  storageBuffer16BitAccess=%d  "
           "uniformAndStorageBuffer16BitAccess=%d\n",
           f16.shaderFloat16, s16.storageBuffer16BitAccess,
           s16.uniformAndStorageBuffer16BitAccess);
    printf("    subgroupSize=%u  subgroupOps=0x%x (arith=%d shuffle=%d)\n",
           sg.subgroupSize, sg.supportedOperations,
           !!(sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT),
           !!(sg.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT));
    printf("    maxComputeSharedMemory=%u KB  maxWG=%u,%u,%u  inv=%u\n",
           p->limits.maxComputeSharedMemorySize / 1024,
           p->limits.maxComputeWorkGroupSize[0],
           p->limits.maxComputeWorkGroupSize[1],
           p->limits.maxComputeWorkGroupSize[2],
           p->limits.maxComputeWorkGroupInvocations);
    printf("    timestampPeriod=%.3f ns  computeAndGraphics=%d\n",
           p->limits.timestampPeriod,
           p->limits.timestampComputeAndGraphics);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, NULL);
    VkQueueFamilyProperties qf[16];
    if(qn > 16) qn = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf);
    for(uint32_t q = 0; q < qn; q++)
      printf("    queue[%u]: flags=0x%x count=%u tsBits=%u%s\n",
             q, qf[q].queueFlags, qf[q].queueCount,
             qf[q].timestampValidBits,
             (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) ? "  <COMPUTE>" : "");
  }
  vkDestroyInstance(inst, NULL);
  return 0;
}
