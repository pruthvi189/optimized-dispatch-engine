import { memo, useMemo } from "react";
import { List } from "react-window";
import type { RowComponentProps } from "react-window";
import type { EventRow } from "../types";
import "./VirtualizedEventLog.css";

const ROW_HEIGHT = 24;
const OVERSIZE = 5;

const toneForEvent = (eventType: string): string => {
  if (eventType.includes("cancelled")) return "bad";
  if (eventType.includes("completed") || eventType.includes("delivered")) return "good";
  return "";
};

interface RowData {
  events: EventRow[];
}

function EventRowItem({ index, style, events }: RowComponentProps<RowData>) {
  const e = events[index];
  const tone = toneForEvent(e.event_type);
  return (
    <div className={`vlog-row ${tone}`} style={style}>
      <span className="vlog-time">{e.sim_time.toFixed(0)}</span>
      <span className="vlog-type">{e.event_type}</span>
      {e.order_id != null && <span className="vlog-ref">order #{e.order_id}</span>}
      {e.rider_id != null && <span className="vlog-ref">rider r{e.rider_id}</span>}
    </div>
  );
}

const VirtualizedEventLog = memo(function VirtualizedEventLog({ events }: { events: EventRow[] }) {
  const reversedEvents = useMemo(() => [...events].reverse(), [events]);
  const rowProps = useMemo(() => ({ events: reversedEvents }), [reversedEvents]);

  if (!events.length) {
    return <div className="vlog-empty">no events yet</div>;
  }

  return (
    <div className="vlog-container">
      <List
        rowCount={reversedEvents.length}
        rowHeight={ROW_HEIGHT}
        rowComponent={EventRowItem}
        rowProps={rowProps}
        overscanCount={OVERSIZE}
        style={{ height: 320, width: "100%" }}
      />
    </div>
  );
});

export { VirtualizedEventLog as EventLog };