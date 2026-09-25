/*
  Round 11 item E: a multi-device composite for the NEW Backend API.

  `multiplexing`, `demux` and `roundrobin` are registered with REGISTER_NETWORK
  and resolve sub-backends through NetworkFactory, which lists only old-API
  Networks. `lc0ex-cuda` is registered with REGISTER_BACKEND and is the only
  new-API Backend in the tree, so it cannot be placed under any of them --
  `--backend=multiplexing --backend-opts="g0(backend=lc0ex-cuda,...)"` fails
  with `Unknown backend: lc0ex-cuda`. lc0ex is therefore single-GPU-per-process,
  which datagen never noticed (one process per GPU) and which a TCEC-style host
  cannot live with.

  The reverse shim -- a Backend wrapped as a Network -- cannot be written:
  NetworkComputation::AddInput receives already-encoded InputPlanes while
  Backend::AddInput needs an EvalPosition (position history plus legal moves),
  and a Position cannot be recovered from planes. So the composite has to live
  on the new side, which is what this is.

  Two distribution modes, because they are different experiments:

    roundrobin (default) -- a whole gather goes to one device, devices taken in
      turn. Per-device batch stays at the search's minibatch size, which is what
      lc0ex's compiled rung ladder is built for, and it is the model for
      GPUs+1 search threads: each thread's batch lands whole on one GPU.

    split -- one gather is divided across all devices and they run in parallel.
      Lower latency for a single large batch, but it divides the batch by the
      device count, which on a rung ladder moves every evaluation to a smaller
      and less efficient program.

  Options are a per-device template rather than one subdict per device, because
  lc0 hands a BackendFactory the TOP-LEVEL options and re-parses `backend-opts`
  from a string, and OptionsDict has no serialiser to rebuild that string from a
  subdict. `;` separates the template's own options so the outer comma-separated
  parse stays unambiguous:

    --backend=multi --backend-opts="backend=lc0ex-cuda,gpus=0:1:2:3,\
        mode=roundrobin,opts=lc0ex=/path/art.lc0ex;concurrency=4;graph=dag"

  which builds, for GPU g, a child whose backend-opts is
  `gpu=g,lc0ex=/path/art.lc0ex,concurrency=4,graph=dag`.
*/

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/optionsdict.h"

namespace lczero {
namespace {

const char* kBackendName = "multi";

enum class Distribution { kRoundRobin, kSplit };

std::vector<std::string> SplitOn(const std::string& value, char separator) {
  std::vector<std::string> parts;
  std::string current;
  for (const char c : value) {
    if (c == separator) {
      if (!current.empty()) parts.push_back(current);
      current.clear();
    } else {
      current += c;
    }
  }
  if (!current.empty()) parts.push_back(current);
  return parts;
}

class MultiBackendComputation final : public BackendComputation {
 public:
  MultiBackendComputation(
      std::vector<std::unique_ptr<BackendComputation>> subs,
      Distribution distribution, size_t start_device)
      : subs_(std::move(subs)),
        distribution_(distribution),
        cursor_(subs_.empty() ? 0 : start_device % subs_.size()) {}

  size_t UsedBatchSize() const override { return used_; }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    const AddInputResult status = subs_[cursor_]->AddInput(pos, result);
    if (status == ENQUEUED_FOR_EVAL) {
      ++used_;
      // Only `split` moves the cursor per input; `roundrobin` keeps a whole
      // gather on one device and advances once per computation instead.
      if (distribution_ == Distribution::kSplit) {
        cursor_ = (cursor_ + 1) % subs_.size();
      }
    }
    return status;
  }

  void ComputeBlocking() override {
    if (used_ == 0) return;
    if (distribution_ == Distribution::kRoundRobin) {
      subs_[cursor_]->ComputeBlocking();
      return;
    }
    // Run every non-empty device concurrently, keeping one on this thread so a
    // single-device configuration spawns no thread at all.
    std::vector<std::thread> workers;
    workers.reserve(subs_.size());
    size_t inline_index = subs_.size();
    for (size_t i = 0; i < subs_.size(); ++i) {
      if (subs_[i]->UsedBatchSize() == 0) continue;
      if (inline_index == subs_.size()) {
        inline_index = i;
        continue;
      }
      workers.emplace_back([this, i] { subs_[i]->ComputeBlocking(); });
    }
    if (inline_index < subs_.size()) subs_[inline_index]->ComputeBlocking();
    for (auto& worker : workers) worker.join();
  }

 private:
  std::vector<std::unique_ptr<BackendComputation>> subs_;
  Distribution distribution_;
  size_t cursor_ = 0;
  size_t used_ = 0;
};

class MultiBackend final : public Backend {
 public:
  explicit MultiBackend(const OptionsDict& options) {
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));

    child_backend_ = backend_options.Get<std::string>("backend");
    if (child_backend_ == kBackendName) {
      throw Exception("The multi backend cannot contain itself.");
    }
    const std::string mode =
        backend_options.GetOrDefault<std::string>("mode", "roundrobin");
    if (mode == "roundrobin") {
      distribution_ = Distribution::kRoundRobin;
    } else if (mode == "split") {
      distribution_ = Distribution::kSplit;
    } else {
      throw Exception("Unknown multi backend mode '" + mode +
                      "'; expected roundrobin or split.");
    }

    const std::vector<std::string> gpus = SplitOn(
        backend_options.GetOrDefault<std::string>("gpus", "0"), ':');
    if (gpus.empty()) throw Exception("The multi backend needs a gpus list.");
    std::string tail =
        backend_options.GetOrDefault<std::string>("opts", std::string());
    std::replace(tail.begin(), tail.end(), ';', ',');
    backend_options.CheckAllOptionsRead(kBackendName);

    for (const std::string& gpu : gpus) {
      std::string device_options = "gpu=" + gpu;
      if (!tail.empty()) device_options += "," + tail;
      device_options_.push_back(device_options);
      OptionsDict child(&options);
      child.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                             device_options);
      devices_.push_back(
          BackendManager::Get()->CreateFromName(child_backend_, child));
    }

    attributes_ = devices_.front()->GetAttributes();
    for (const auto& device : devices_) {
      const BackendAttributes other = device->GetAttributes();
      attributes_.has_mlh = attributes_.has_mlh && other.has_mlh;
      attributes_.has_wdl = attributes_.has_wdl && other.has_wdl;
      attributes_.runs_on_cpu = attributes_.runs_on_cpu && other.runs_on_cpu;
      // A whole gather must fit whichever device receives it.
      attributes_.maximum_batch_size =
          std::min(attributes_.maximum_batch_size, other.maximum_batch_size);
      attributes_.recommended_batch_size = std::min(
          attributes_.recommended_batch_size, other.recommended_batch_size);
    }
    if (distribution_ == Distribution::kSplit) {
      // A split gather is divided, so the composite can accept device-count
      // times as much and each device still sees a batch it can run.
      attributes_.maximum_batch_size *= static_cast<int>(devices_.size());
      attributes_.recommended_batch_size *= static_cast<int>(devices_.size());
    }
    // GPUs + 1, which is the lab's standing rule for a multiplexed run.
    attributes_.suggested_num_search_threads =
        static_cast<int>(devices_.size()) + 1;
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    std::vector<std::unique_ptr<BackendComputation>> subs;
    subs.reserve(devices_.size());
    for (auto& device : devices_) subs.push_back(device->CreateComputation());
    // Each new computation starts on the next device, so gathers from different
    // search threads land on different GPUs.
    const size_t start = next_device_.fetch_add(1);
    return std::make_unique<MultiBackendComputation>(std::move(subs),
                                                     distribution_, start);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    UpdateConfigurationResult result = UPDATE_OK;
    for (size_t i = 0; i < devices_.size(); ++i) {
      // Each device must see ITS OWN backend-opts string, or it would compare
      // the composite's against the one it was built with and demand a restart
      // on every configuration update.
      OptionsDict child(&options);
      child.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                             device_options_[i]);
      if (devices_[i]->UpdateConfiguration(child) == NEED_RESTART) {
        result = NEED_RESTART;
      }
    }
    return result;
  }

 private:
  std::vector<std::unique_ptr<Backend>> devices_;
  std::vector<std::string> device_options_;
  std::string child_backend_;
  Distribution distribution_ = Distribution::kRoundRobin;
  BackendAttributes attributes_;
  std::atomic<size_t> next_device_{0};
};

class MultiBackendFactory final : public BackendFactory {
 public:
  // Below every real backend, so autodiscovery never selects it.
  int GetPriority() const override { return -2000; }
  std::string_view GetName() const override { return kBackendName; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<MultiBackend>(options);
  }
};

REGISTER_BACKEND(MultiBackendFactory)

}  // namespace
}  // namespace lczero
