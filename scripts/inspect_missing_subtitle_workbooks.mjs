import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const targets = [
  ["output/log/translation_results.xlsx", "output/log/translation_results_before_repair.png", "A458:H470"],
  ["output/log/cleaned_chunks.xlsx", "output/log/cleaned_chunks_before_repair.png", "A4175:D4190"],
];

for (const [inputPath, previewPath, range] of targets) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  const sheet = workbook.worksheets.getItemAt(0);
  const summary = await workbook.inspect({
    kind: "sheet,table",
    maxChars: 2500,
    tableMaxRows: 4,
    tableMaxCols: 10,
  });
  console.log(inputPath, summary.ndjson);
  const preview = await workbook.render({ sheetName: sheet.name, range, scale: 1.5, format: "png" });
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
}
