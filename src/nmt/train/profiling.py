# Opt-in wrappers around torch.profiler (dataloader vs attention vs FFN bottlenecks)
# torch.cuda.memory_summary (OOM debugging at IITB scale).

import torch

def make_profiler(log_dir, wait=1, warmup=1, active=3, repeat=1):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    on_trace_ready = torch.profiler.tensorboard_trace_handler(log_dir)
    profiler = torch.profiler.profile(activities=activities, schedule=schedule, on_trace_ready=on_trace_ready, profile_memory = True, with_stack = False, record_shapes=True)
    return profiler

def print_summary(prof, sort_by=None, row_limit=20):
    # prof.key_averages()
    # .table(sort_by=sort_by, row_limit=row_limit)
    if sort_by is None:
        sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))

def cuda_memory_summary(device=None):
    if torch.cuda.is_available() is False:
        return "no CUDA device"
    return torch.cuda.memory_summary(device)

def reset_peak_memory(device=None):
    if torch.cuda.is_available() is False:
        return "no CUDA device"
    else:
        torch.cuda.reset_peak_memory_stats(device)

def peak_memory_mb(device=None):
    if torch.cuda.is_available() is False:
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1000000
