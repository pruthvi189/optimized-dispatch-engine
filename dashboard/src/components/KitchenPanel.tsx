import type { KitchenState } from "../types";

export function KitchenPanel({ kitchens }: { kitchens: KitchenState[] }) {
  const queued = kitchens.reduce((s, k) => s + k.queue_len, 0);
  const inFlight = kitchens.reduce((s, k) => s + k.orders.length, 0);

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Kitchens</h3>
        <span className="panel-meta">
          {kitchens.length
            ? `${kitchens.length} active · ${queued} queued · ${inFlight} in flight`
            : "—"}
        </span>
      </div>
      <div className="panel-body">
        {!kitchens.length ? (
          <div className="empty">not started</div>
        ) : (
          <div className="kitchens">
            {kitchens.map((k) => (
              <div className="kitchen" key={k.id}>
                <div className="khead">
                  <span className="kname">Kitchen {k.id}</span>
                  <span className="qlen">{k.queue_len} queued</span>
                </div>
                {k.orders.length === 0 ? (
                  <div className="empty slim">idle</div>
                ) : (
                  k.orders.map((o) => (
                    <div className="order-row" key={o.id}>
                      <span className="id">#{o.id}</span>
                      <span className="st">{o.status}</span>
                      <span className="st">{o.complexity} ×{o.items}</span>
                      <span className="st">@{o.placed_at.toFixed(0)}</span>
                    </div>
                  ))
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}