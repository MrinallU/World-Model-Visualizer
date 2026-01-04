import React from "react";
import { SliderField } from "./SliderField";


export function SliderGrid({
  styles,

  latent,
  onChangeLatent,

  values,
  onChangeIndex,
  title = "Latent controls",
  description,
  maxHeight = 520,

  indices,
  unusedIndices,
  labelForIndex,
  formatValue,
  rangeForIndex,
}) {
  const arr = values ?? latent ?? [];
  const onChange = onChangeIndex ?? onChangeLatent;

  if (!styles) throw new Error("SliderGrid: missing `styles` prop");
  if (!Array.isArray(arr)) throw new Error("SliderGrid: values/latent must be array");
  if (typeof onChange !== "function") throw new Error("SliderGrid: missing change handler");

  const renderIndices = indices ?? arr.map((_, i) => i);

  const isUnused = (i) => {
    if (!unusedIndices) return false;
    if (unusedIndices instanceof Set) return unusedIndices.has(i);
    if (Array.isArray(unusedIndices)) return unusedIndices.includes(i);
    return false;
  };

  const _labelForIndex = labelForIndex ?? ((i) => `z[${i}]`);
  const _formatValue = formatValue ?? ((v) => Number(v).toFixed(2));
  const _rangeForIndex = rangeForIndex ?? (() => ({ min: -3, max: 3, step: 0.05 }));

  // scrolling heuristic now independent of columns
  const shouldScroll = renderIndices.length > 6;

  return (
    <div style={{ marginTop: 12, borderTop: "1px solid rgba(15,23,42,0.08)", paddingTop: 12 }}>
      {!!title && (
        <div style={{ fontWeight: 800, fontSize: 13.5, color: "#0f172a" }}>
          {title}
        </div>
      )}

      {description ? (
        <div style={styles.smallText}>{description}</div>
      ) : (
        <div style={styles.smallText}>
          Edit <span style={styles.kbd}>z[i]</span> manually to probe what directions the decoder uses.
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          columnGap: 16,
          rowGap: 12,
          ...(shouldScroll
            ? { maxHeight, overflowY: "auto", paddingRight: 6, marginTop: 10 }
            : { marginTop: 10 }),
        }}
      >
        {renderIndices.map((i, renderPos) => {
          const v = arr[i];
          const r = _rangeForIndex(i, v, renderPos) || {};
          const min = r.min ?? -3;
          const max = r.max ?? 3;
          const step = r.step ?? 0.05;
          const width = r.width;

          return (
            <SliderField
              key={i}
              styles={styles}
              label={_labelForIndex(i, v, renderPos)}
              value={v}
              displayValue={_formatValue(v, i, renderPos)}
              min={min}
              max={max}
              step={step}
              width={width}
              unused={isUnused(i)}
              onChange={(e) => onChange(i, Number(e.target.value))}
            />
          );
        })}
      </div>
    </div>
  );
}

