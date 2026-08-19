/**
 * Shared formatting helpers for dispatch dashboard.
 * Keeps formatting logic DRY across components.
 */

/** Format prep estimate with optional range: "7.4 min" or "7.4 min (6.2–8.7 min)" */
export function formatPrepEst(
  prepMean: number | null,
  prepLow: number | null,
  prepHigh: number | null
): string {
  if (prepMean == null) return "—";
  const hasRange = prepLow != null && prepHigh != null && prepLow !== prepHigh;
  return hasRange
    ? prepMean.toFixed(1) + " min (" + prepLow.toFixed(1) + "–" + prepHigh.toFixed(1) + " min)"
    : prepMean.toFixed(1) + " min";
}

/** Format delivery estimate: "24.8 min" or "—" */
export function formatDeliveryEst(
  deliveryEstMin: number | null | undefined,
  totalEstMin?: number | null
): string {
  const val = deliveryEstMin ?? totalEstMin;
  return val != null ? val.toFixed(1) + " min" : "—";
}

/** Format a simple number with unit: "2.2 min" or "—" */
export function formatNumberWithUnit(
  val: number | null | undefined,
  unit: string = "min"
): string {
  return val != null ? val.toFixed(1) + " " + unit : "—";
}