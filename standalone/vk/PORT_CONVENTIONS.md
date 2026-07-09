# OpenCL→GLSL transcription conventions (V2.0 Phase B2)

One `.comp` per kernel in `standalone/vk/shaders/`, filename = kernel name
minus `galosh_` (e.g. `galosh_o32_gat_forward_full` → `o32_gat_forward_full.comp`).

- `#version 450` first line.
- Buffer args (OpenCL `global T *`) → `layout(std430, binding = N) buffer BufN { T name[]; };`
  with binding N = the argument's position among POINTER args (0-based, in
  OpenCL argument order). Mark `readonly`/`writeonly` when the kernel only
  reads/writes it.
- Scalar args → ONE `layout(push_constant) uniform PC { ... } pc;` block,
  fields in OpenCL argument order (int/uint/float are 4 bytes each).
- `local_size_x/y` = the local work size the OpenCL HOST passes for that
  kernel (find the clEnqueueNDRangeKernel for its `k_...` handle in
  `standalone/galosh_raw_gpu.c` and copy it; if NULL local size, choose 256x1
  and note it).
- `__local T x[N]` → `shared T x[N];`
- `barrier(CLK_LOCAL_MEM_FENCE)` → `memoryBarrierShared(); barrier();`
- ID mapping: `get_global_id(0/1)` → `gl_GlobalInvocationID.x/.y`,
  `get_local_id` → `gl_LocalInvocationID`, `get_group_id` → `gl_WorkGroupID`,
  `get_num_groups` → `gl_NumWorkGroups`, `get_local_size(i)` → constant.
- Builtins: `mad(a,b,c)`→`fma`, `rsqrt`→`inversesqrt`, `fabs`→`abs`,
  `native_exp/exp`→`exp`, `convert_int(x)`→`int(x)`, `fmax/fmin`→`max/min`,
  `select(a,b,c)`→`mix`/ternary (careful: OpenCL select picks b when c true).
- Guard pattern: OpenCL kernels early-return on `gid >= n`; keep identical.
- Float atomics (`atomic_*` on float, if any) → note in manifest; use
  GL_EXT_shader_atomic_float and record the requirement.
- NO semantic changes, NO "improvements": transcription must be literal —
  correctness is proven by phase-level readback parity vs the CPU reference.
- 転写は逐語。最適化・整形は B3 で行う。正しさはフェーズ parity で証明する。

Every produced file MUST compile: `glslc shaders/<f>.comp -o /dev/null`
(PATH: C:\msys64\ucrt64\bin). Fix until clean.

Per kernel also record a manifest entry (JSON, in your group file
`standalone/vk/manifest_<group>.json`):
```json
{ "kernel": "galosh_o32_...", "comp": "o32_....comp",
  "bindings": [{"b":0,"arg":"in_gat","dir":"r"}, ...],
  "push": [{"n":"width","t":"int"}, ...],
  "local": [16,16], "dispatch": "<quote the host's global-size computation>",
  "notes": "float atomics / shared mem bytes / anything nonstandard" }
```
