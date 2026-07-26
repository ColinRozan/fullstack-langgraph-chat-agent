import html2canvas from "html2canvas";
import jsPDF from "jspdf";

/**
 * Download a string as a file
 */
function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Export text content as a Markdown file
 */
export function exportAsMarkdown(content: string, filename?: string) {
  const safeName = filename || `answer-${Date.now()}.md`;
  const finalName = safeName.endsWith(".md") ? safeName : `${safeName}.md`;
  downloadFile(content, finalName, "text/markdown");
}

/**
 * Export text content as a plain text file
 */
export function exportAsText(content: string, filename?: string) {
  const safeName = filename || `answer-${Date.now()}.txt`;
  const finalName = safeName.endsWith(".txt") ? safeName : `${safeName}.txt`;
  downloadFile(content, finalName, "text/plain");
}

/**
 * Export a DOM element as PDF using html2canvas + jsPDF
 */
export async function exportAsPDF(
  element: HTMLElement,
  filename?: string
) {
  const safeName = filename || `answer-${Date.now()}.pdf`;
  const finalName = safeName.endsWith(".pdf") ? safeName : `${safeName}.pdf`;

  const canvas = await html2canvas(element, {
    scale: 2,
    useCORS: true,
    backgroundColor: "#171717", // matches neutral-900 roughly
    logging: false,
  });

  const imgData = canvas.toDataURL("image/png");

  // A4 page dimensions in mm
  const pageWidth = 210;
  const pageHeight = 297;
  const margin = 10;
  const usableWidth = pageWidth - margin * 2;

  const imgWidth = canvas.width;
  const imgHeight = canvas.height;
  const ratio = Math.min(usableWidth / imgWidth, 1);
  const scaledWidth = imgWidth * ratio;
  const scaledHeight = imgHeight * ratio;

  const pdf = new jsPDF("p", "mm", "a4");

  let heightLeft = scaledHeight;
  let position = 0;

  // First page
  pdf.addImage(imgData, "PNG", margin, margin, scaledWidth, scaledHeight);
  heightLeft -= pageHeight - margin * 2;

  // Additional pages if content overflows
  while (heightLeft > 0) {
    position = heightLeft - scaledHeight + margin;
    pdf.addPage();
    pdf.addImage(imgData, "PNG", margin, position, scaledWidth, scaledHeight);
    heightLeft -= pageHeight - margin * 2;
  }

  pdf.save(finalName);
}
