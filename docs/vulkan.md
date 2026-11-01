# Linux Vulkan port

`feat/vulkan` is the current Splash tree plus the recovered M5/Linux Vulkan
backend.  It preserves the latest engine and Metal sources and adds the
Vulkan implementation under `runtime/vulkan/`.

The qualification source was the Splash checkout at `/opt/splash-linux-vulkan`
on the M5.  Its serving entry points were:

```text
/opt/splash-linux-vulkan/dev/serve-m5-vulkan
/opt/splash-linux-vulkan/dev/splash-m5-vulkan
```

The recovered implementation includes:

- `runtime/vulkan/VulkanBackend.cpp`, implementing the existing backend
  contract with Vulkan buffer management, pipeline caching, timestamps,
  command submission, and optional shader-object imports;
- the Vulkan compute shaders for GGUF repacking, registered Q5/Q6
  projections, attention, GDN, draft selection, and DFlash verification;
- the G17G Q5/Q6 native shader-object package and the Metal source used to
  produce it;
- `runtime/vulkan/abi/Linear.h`, the scalar host/shader parameter contract.

The exact target used for the reported qualification was the unchanged
Qwen3.8-27B Q6_K GGUF with SHA-256
`c9c206812fbe4ac7b76a729e25928b63f2ae89d37f69da7a71c20aec763cd436`.
The R354b qualification recorded 50.04 tok/s on the short coding workload
and 40.90 tok/s on the long prompt.  Those numbers are tied to the driver
package recorded in the lab handoff; they are not a promise that a different
driver will reproduce them.

## Rebase boundary

The Vulkan checkout predates the latest engine and storage refactors on
`main`.  For that reason the port is isolated from the default Makefile and
macOS build in this first publication.  The next integration step is to wire
the current `runtime/engine`, model, and Linux platform shims into a CMake
Vulkan build, then run the Vulkan numerical and serving qualification on the
M5.  No Metal default is changed by this branch.

The source and qualification receipts remain in the local lab archive at
`/home/dj/Work/m5-splash-opt-r78`.
