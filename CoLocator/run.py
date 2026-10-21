import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
import signal
import sys
import os
import pickle

active_processes = []
def run_instance(i, args):
    print(f"🚀 Starting Instance {i+1}...")
    
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "main.py",
                "-d", args.d,
                "-s", args.s,
                "-i", str(i),
                "-m", args.m,
                "-limit_loc", str(args.limit_loc),
                "-limit_kg", str(args.limit_kg),
                "-seed", str(args.seed)
            ],
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        active_processes.append(proc)
        proc.wait()
        
        if proc.returncode == 0:
            return f"✅ Instance {i+1}: SUCCESS"
        else:
            return f"❌ Instance {i+1}: FAILED (Code {proc.returncode})"
    except Exception as e:
        return f"⚠️ Instance {i+1}: ERROR ({str(e)})"

def signal_handler(sig, frame):
    print("\n\n🛑 [STOP] Ctrl+C detected! Cleaning up processes...")
    for proc in active_processes:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                print(f"--- Terminated process {proc.pid}")
            except:
                proc.terminate()
    
    print("✨ Cleanup complete. Exiting.")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live")
    parser.add_argument("-s", type=str, default="lite")
    parser.add_argument("-n", type=int, default=600)
    parser.add_argument("-m", type=str, default="qwen2.5")
    parser.add_argument("-w", "--workers", type=int, default=20)
    parser.add_argument("-limit_kg", type=int, default=10)
    parser.add_argument("-limit_loc", type=int, default=60)
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    print(f"🔥 Parallel execution started (Workers: {args.workers}). Press Ctrl+C to stop all.\n")

    name_data = args.d.strip().split("/")[1]
    with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
        data = pickle.load(f)
    DIR_RES = f"{name_data}/{args.m}_loc_{args.limit_loc}_kg_{args.limit_kg}_seed_{args.seed}"

    tasks = []
    for i in range(min(args.n, len(data))):
        instance = data[i]["instance_id"]
        if not os.path.exists(f"./res/{DIR_RES}/results_{instance}.json"):
            tasks.append(i)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        try:
            futures = [executor.submit(run_instance, i, args) for i in tasks]
            for future in futures:
                try:
                    print(future.result())
                except Exception:
                    pass
        except KeyboardInterrupt:
            signal_handler(None, None)

if __name__ == "__main__":
    main()