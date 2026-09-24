import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const translationPath = "output/log/translation_results.xlsx";
const cleanedPath = "output/log/cleaned_chunks.xlsx";
const translationBackup = "output/log/translation_results.before_missing_subtitle_repair.xlsx";
const cleanedBackup = "output/log/cleaned_chunks.before_missing_subtitle_repair.xlsx";

const replacements = [
  {
    source: "and seeing an engine you could understand, diagnose, and fix",
    translation: "眼前的发动机结构一目了然，能够理解、诊断，",
    start: 1801.518,
    end: 1806.440,
  },
  {
    source: "with your own two hands and your own hard-won experience.",
    translation: "凭自己的双手和多年积累的经验就能修好。",
    start: 1806.440,
    end: 1810.400,
  },
  {
    source: "Just iron, fuel, and fire doing exactly what they were built to do.",
    translation: "不过是钢铁、燃油与烈火，各尽其职。",
    start: 1810.400,
    end: 1816.360,
  },
  {
    source: "Those three brands didn't just build engines. They built the culture of American trucking.",
    translation: "这三个品牌造的不只是发动机，更塑造了美国卡车文化。",
    start: 1816.360,
    end: 1821.160,
  },
  {
    source: "They gave drivers something to argue about, something to be loyal to,",
    translation: "它们让司机有了争论的话题，也有了忠于某个品牌的理由，",
    start: 1821.160,
    end: 1825.560,
  },
  {
    source: "something to identify with beyond the freight and the miles.",
    translation: "在货物与里程之外，也有了自己的身份认同。",
    start: 1825.560,
    end: 1829.440,
  },
  {
    source: "And those arguments",
    translation: "而这些争论，",
    start: 1829.440,
    end: 1830.729,
  },
];

function srtTime(seconds) {
  const milliseconds = Math.round(seconds * 1000);
  const hours = Math.floor(milliseconds / 3600000);
  const minutes = Math.floor((milliseconds % 3600000) / 60000);
  const secs = Math.floor((milliseconds % 60000) / 1000);
  const millis = milliseconds % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(millis).padStart(3, "0")}`;
}

function translationRow(item) {
  return [
    item.source,
    item.translation,
    `${srtTime(item.start)} --> ${srtTime(item.end)}`,
    item.end - item.start,
    false,
    null,
    "manual_source_repair",
    item.translation,
  ];
}

function wordRows(item) {
  const words = item.source.split(/\s+/).filter(Boolean);
  const step = (item.end - item.start) / words.length;
  return words.map((word, index) => [
    `"${word}"`,
    item.start + index * step,
    item.start + (index + 1) * step,
    null,
  ]);
}

async function loadWorkbook(path) {
  return SpreadsheetFile.importXlsx(await FileBlob.load(path));
}

async function saveWorkbook(workbook, path) {
  workbook.recalculate();
  const exported = await SpreadsheetFile.exportXlsx(workbook);
  await exported.save(path);
}

try { await fs.access(translationBackup); } catch { await fs.copyFile(translationPath, translationBackup); }
try { await fs.access(cleanedBackup); } catch { await fs.copyFile(cleanedPath, cleanedBackup); }

const translationWorkbook = await loadWorkbook(translationPath);
const translationSheet = translationWorkbook.worksheets.getItemAt(0);
const translationValues = translationSheet.getUsedRange(true).values;
const badTranslationIndex = translationValues.findIndex(
  (row, index) => index > 0 && row[0] === "Thank you for watching."
);
if (badTranslationIndex < 0) {
  throw new Error("Could not find the erroneous translation row.");
}
const repairedTranslationValues = [
  ...translationValues.slice(0, badTranslationIndex),
  ...replacements.map(translationRow),
  ...translationValues.slice(badTranslationIndex + 1),
];
translationSheet.getUsedRange().clear({ applyTo: "contents" });
translationSheet.getRange("A1").write(repairedTranslationValues);
await saveWorkbook(translationWorkbook, `${translationPath}.part.xlsx`);

const cleanedWorkbook = await loadWorkbook(cleanedPath);
const cleanedSheet = cleanedWorkbook.worksheets.getItemAt(0);
const cleanedValues = cleanedSheet.getUsedRange(true).values;
const badWordStart = cleanedValues.findIndex(
  (row, index) => index > 0 && row[0] === '"Thank"' && Number(row[1]) > 1800
);
if (badWordStart < 0) {
  throw new Error("Could not find the erroneous cleaned-word range.");
}
const badWordEnd = badWordStart + 4;
const insertedWords = replacements.flatMap(wordRows);
const repairedCleanedValues = [
  ...cleanedValues.slice(0, badWordStart),
  ...insertedWords,
  ...cleanedValues.slice(badWordEnd),
];
cleanedSheet.getUsedRange().clear({ applyTo: "contents" });
cleanedSheet.getRange("A1").write(repairedCleanedValues);
await saveWorkbook(cleanedWorkbook, `${cleanedPath}.part.xlsx`);

console.log(JSON.stringify({
  translationRowsBefore: translationValues.length - 1,
  translationRowsAfter: repairedTranslationValues.length - 1,
  cleanedRowsBefore: cleanedValues.length - 1,
  cleanedRowsAfter: repairedCleanedValues.length - 1,
  insertedTranslationRows: replacements.length,
  insertedWordRows: insertedWords.length,
}));
