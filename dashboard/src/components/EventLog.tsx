import type { EventRow } from "../types";

const toneForEvent = (eventType: string): string => {
  if (eventType.includes("cancelled")) return "bad";
  if (eventType.includes("completed") || eventType.includes("delivered")) return "good";
  return "";
};

export function EventLog({ events }: { events: EventRow[] }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Event stream</h3>
        <span className="panel-meta">
          {events.length ? `last ${events.length} events` : "—"}
        </span>
      </div>
      <div className="panel-body">
        <div className="log">
          {!events.length ? (
            <div className="empty">no events yet</div>
          ) : (
            [...events].reverse().map((e, i) => (
              <div className={`row ${toneForEvent(e.event_type)}`} key={`${e.sim_time}-${e.order_id}-${e.rider_id}-${i}`}>
                <span className="t">{e.sim_time.toFixed(0)}</span>
                <span className="type">{e.event_type}</span>
                {e.order_id != null && <span className="ref">order #{e.order_id}</span>}
                {e.rider_id != null && <span className="ref">rider r{e.rider_id}</span>}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}