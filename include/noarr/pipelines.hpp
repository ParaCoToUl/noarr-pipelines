/*
    This file aggregates includes for the entire "pipelines" module
 */

// common utilities
#include "noarr/pipelines/Device.hpp"

// memory management layer
#include "noarr/pipelines/Buffer.hpp"
#include "noarr/pipelines/MemoryAllocator.hpp"
#include "noarr/pipelines/MemoryTransferer.hpp"
#include "noarr/pipelines/HostAllocator.hpp"
#include "noarr/pipelines/DummyGpuAllocator.hpp"
#include "noarr/pipelines/HostTransferer.hpp"
#include "noarr/pipelines/HardwareManager.hpp"

// pipeline basics
#include "noarr/pipelines/UntypedEnvelope.hpp"
#include "noarr/pipelines/Envelope.hpp"
#include "noarr/pipelines/Link.hpp"
#include "noarr/pipelines/Node.hpp"

// pipeline computation
#include "noarr/pipelines/ComputeNode.hpp"
#include "noarr/pipelines/AsyncComputeNode.hpp"
#include "noarr/pipelines/LambdaComputeNode.hpp"
#include "noarr/pipelines/LambdaAsyncComputeNode.hpp"

// pipeline data
#include "noarr/pipelines/Hub.hpp"

// scheduling
#include "noarr/pipelines/Scheduler.hpp"
#include "noarr/pipelines/DebuggingScheduler.hpp"
#include "noarr/pipelines/SimpleScheduler.hpp"
