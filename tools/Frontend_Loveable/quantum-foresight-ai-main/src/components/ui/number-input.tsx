import { useState, useEffect, type ComponentProps } from "react";
import { Input } from "@/components/ui/input";

interface NumberInputProps extends Omit<ComponentProps<typeof Input>, "value" | "onChange" | "type"> {
  value: number;
  onChange: (value: number) => void;
  /** Restored on blur when the field is left empty/invalid. Defaults to 0. */
  fallback?: number;
  min?: number;
}

/**
 * A type="number" input that keeps its own text buffer so it never fights
 * the user's typing — deleting down to an empty field just stays empty
 * instead of snapping back to the last numeric value. The parent's onChange
 * only fires while the text parses to a real number; on blur, an empty or
 * invalid field resets to `fallback` (and calls onChange with it).
 */
export function NumberInput({ value, onChange, fallback = 0, min, ...props }: NumberInputProps) {
  const [text, setText] = useState(String(value));

  useEffect(() => {
    // Only resync from the parent when it reflects a genuinely external
    // change — otherwise this would clobber things like a trailing "." the
    // user just typed, since that keystroke's own onChange already pushed
    // the coerced number back up to the parent.
    if (parseFloat(text) !== value) {
      setText(String(value));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <Input
      {...props}
      type="number"
      min={min}
      value={text}
      onChange={(e) => {
        const raw = e.target.value;
        setText(raw);
        const parsed = parseFloat(raw);
        if (!Number.isNaN(parsed)) {
          onChange(min !== undefined ? Math.max(min, parsed) : parsed);
        }
      }}
      onBlur={(e) => {
        const parsed = parseFloat(text);
        const finalValue = Number.isNaN(parsed) ? fallback : (min !== undefined ? Math.max(min, parsed) : parsed);
        setText(String(finalValue));
        onChange(finalValue);
        props.onBlur?.(e);
      }}
    />
  );
}
