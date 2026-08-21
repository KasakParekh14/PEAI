import Papa from "papaparse";
import * as XLSX from "xlsx";

export type Row = Record<string, number | string | null>;
export interface ParsedDataset {
  rows: Row[];
  columns: string[];
  numericColumns: string[];
}

export async function parseFile(file: File): Promise<ParsedDataset> {
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (ext === "xlsx" || ext === "xls") {
    const buf = await file.arrayBuffer();
    const wb = XLSX.read(buf, { type: "array" });
    const sheet = wb.Sheets[wb.SheetNames[0]];
    const json = XLSX.utils.sheet_to_json<Row>(sheet, { defval: null });
    return enrich(json);
  }
  const text = await file.text();
  const result = Papa.parse<Row>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    delimiter: ext === "txt" ? undefined : ",",
  });
  return enrich(result.data);
}

function enrich(rows: Row[]): ParsedDataset {
  const columns = rows.length ? Object.keys(rows[0]) : [];
  const numericColumns = columns.filter((c) =>
    rows.slice(0, 50).every((r) => r[c] === null || typeof r[c] === "number")
  );
  return { rows, columns, numericColumns };
}

export function detectRange(rows: Row[], col: string) {
  const vals = rows.map((r) => Number(r[col])).filter((v) => Number.isFinite(v));
  return { min: Math.min(...vals), max: Math.max(...vals), count: vals.length };
}

export function cleanNumeric(rows: Row[], xCol: string, yCol: string) {
  return rows
    .map((r) => ({ x: Number(r[xCol]), y: Number(r[yCol]) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => a.x - b.x);
}
