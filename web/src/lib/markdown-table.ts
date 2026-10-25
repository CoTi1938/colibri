/* GFM pipe tables, parsed to plain data so the renderer stays a pure mapping.
 *
 * Kept out of the component on purpose: this is the part with the edge cases
 * (escaped pipes, optional outer pipes, ragged rows), and it is testable here
 * without a DOM. The renderer turns the result into React nodes, so cell text
 * never reaches innerHTML. */

export type Align = "left" | "center" | "right" | null

export type Table = {
  head: string[]
  align: Align[]
  rows: string[][]
  /** Lines consumed, so the block loop can skip past the table. */
  length: number
}

/* Split one row into cells. The outer pipes are delimiters when present, so
   they are stripped before splitting rather than left to produce empty edge
   cells — otherwise "| a | b |" and "a | b" would disagree on column count.
   A pipe is content only when an odd run of backslashes precedes it. An even
   run is that many literal backslashes, and the pipe is still a delimiter:
   `\\|` is a path that ends in `\` followed by the next cell. */
export function splitRow(line: string): string[] {
  let text = line.trim()
  if (text.startsWith("|")) text = text.slice(1)
  if (text.endsWith("|")) {
    let slashes = 0
    for (let i = text.length - 2; i >= 0 && text[i] === "\\"; i--) slashes++
    if (slashes % 2 === 0) text = text.slice(0, -1)
  }

  const out: string[] = []
  let cell = ""
  for (let at = 0; at < text.length; at++) {
    if (text[at] === "\\") {
      let run = 1
      while (text[at + run] === "\\") run++
      cell += "\\".repeat(Math.floor(run / 2))
      if (run % 2 === 1 && text[at + run] === "|") {
        cell += "|"
        at += run
        continue
      }
      if (run % 2 === 1) cell += "\\"
      at += run - 1
      continue
    }
    if (text[at] === "|") { out.push(cell); cell = ""; continue }
    cell += text[at]
  }
  out.push(cell)
  return out.map((item) => item.trim())
}

/* The delimiter row is what makes a run of pipes a table: every cell has to be
   dashes, optionally colon-anchored. Anything else and this was ordinary text
   that happened to contain a pipe. */
export function rowAlignments(line: string): Align[] | null {
  if (!line || !line.includes("-")) return null
  const parts = splitRow(line)
  if (!parts.length) return null

  const out: Align[] = []
  for (const part of parts) {
    const match = /^(:?)-+(:?)$/.exec(part)
    if (!match) return null
    const left = match[1] === ":"
    const right = match[2] === ":"
    out.push(left && right ? "center" : right ? "right" : left ? "left" : null)
  }
  return out
}

/* Read a table starting at `from`, or null if there is not one there.
 *
 * Ragged rows are kept as the model wrote them rather than truncated to the
 * header width: a short row is padded so the grid holds, and a long one keeps
 * its extra cells. Dropping them would silently lose content, which is worse
 * than one wide row. */
export function parseTableAt(lines: string[], from: number): Table | null {
  const header = lines[from]
  if (header === undefined || !header.includes("|")) return null

  const align = rowAlignments(lines[from + 1])
  if (!align) return null

  /* GFM: the header and the delimiter row have the same number of cells, or this
     is not a table. A line with a pipe followed by a "---" rule is prose and a rule. */
  const head = splitRow(header)
  if (!head.length || head.every((cell) => !cell) || head.length !== align.length) return null

  const rows: string[][] = []
  let at = from + 2
  for (; at < lines.length; at++) {
    const line = lines[at]
    if (!line.trim() || !line.includes("|")) break
    const row = splitRow(line)
    while (row.length < head.length) row.push("")
    rows.push(row)
  }

  return { head, align, rows, length: at - from }
}
