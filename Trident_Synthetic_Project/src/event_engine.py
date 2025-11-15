
import heapq, time, argparse, os
import pandas as pd

class EventEngine:
    """Priority queue based event engine.
    Events are tuples: (priority, timestamp, event_type, data)
    Lower 'priority' values are processed first.
    """
    def __init__(self):
        self._queue = []
        self._handlers = {}

    def add_handler(self, event_type, fn):
        self._handlers[event_type] = fn

    def add_event(self, priority, timestamp, event_type, data=None):
        heapq.heappush(self._queue, (priority, timestamp, event_type, data))

    def run(self, max_events=None):
        processed = 0
        start = time.time()
        while self._queue and (max_events is None or processed < max_events):
            priority, ts, ev_type, data = heapq.heappop(self._queue)
            handler = self._handlers.get(ev_type)
            if handler:
                handler(ts, data)
            processed += 1
        duration = time.time() - start
        return processed, duration

# Example handlers
def handle_threshold(ts, data):
    # lightweight processing: print or log
    # In real systems this could trigger actions, logging, or downstream computation
    print(f"[THRESHOLD] {ts} -> power={data['power']:.2f}, diff={data['diff']:.3f}")

def build_events_from_data(df, engine, power_diff_threshold=20.0):
    # Create events where absolute change in power from previous sample exceeds threshold
    prev = None
    for idx, row in df.iterrows():
        if prev is not None:
            diff = abs(row['power'] - prev['power'])
            if diff >= power_diff_threshold:
                # priority: larger diffs get higher priority (smaller number)
                priority = int(max(1, 100 - diff))
                engine.add_event(priority, row['timestamp'], 'threshold', {'power': row['power'], 'diff': diff})
        prev = row
    return engine

def demo(data_csv='"C:\\Users\\DHANASHRI\\Downloads\\Trident_Synthetic_Project\\data\\synthetic_timeseries.csv"', limit_events=50):
    df = pd.read_csv(data_csv, parse_dates=['timestamp'])
    engine = EventEngine()
    engine.add_handler('threshold', handle_threshold)
    engine = build_events_from_data(df, engine, power_diff_threshold=20.0)
    processed, duration = engine.run(max_events=limit_events)
    print(f"Processed {processed} events in {duration:.3f} s (demo limit={limit_events})")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='C:\\Users\\DHANASHRI\\Downloads\\Trident_Synthetic_Project\\data\\synthetic_timeseries.csv')
    p.add_argument('--limit', type=int, default=50)
    args = p.parse_args()
    demo(data_csv=args.data, limit_events=args.limit)
