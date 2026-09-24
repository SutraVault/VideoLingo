import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const targets = [
  ["output/log/translation_results.xlsx", "A460:H474", "output/log/translation_results_after_repair.png"],
  ["output/log/cleaned_chunks.xlsx", "A4177:D4210", "output/log/cleaned_chunks_after_repair.png"],
];

for (const [path, range, previewPath] of targets) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const sheet = workbook.worksheets.getItemAt(0);
  const check = await workbook.inspect({
    kind: "table",
    sheetId: sheet.name,
    range,
    include: "values,formulas",
    tableMaxRows: 40,
    tableMaxCols: 10,
    maxChars: 12000,
  });
  console.log(path, check.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 50 },
    summary: "final formula error scan",
  });
  console.log("errors", errors.ndjson);
  const preview = await workbook.render({ sheetName: sheet.name, range, scale: 1.5, format: "png" });
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
}
