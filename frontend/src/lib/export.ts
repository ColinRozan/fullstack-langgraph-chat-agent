import * as htmlToImage from "html-to-image";
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
 * Export a DOM element as PDF using html-to-image + jsPDF.
 * Renders in a light-theme clone so the PDF is white-background / dark-text.
 */
export async function exportAsPDF(
  element: HTMLElement,
  filename?: string
) {
  const safeName = filename || `answer-${Date.now()}.pdf`;
  const finalName = safeName.endsWith(".pdf") ? safeName : `${safeName}.pdf`;

  // Clone the element into an on-screen but invisible wrapper
  // so html-to-image can measure and render it correctly.
  const clone = element.cloneNode(true) as HTMLElement;
  const wrapper = document.createElement("div");
  wrapper.style.background = "#ffffff";
  wrapper.style.color = "#171717";
  wrapper.style.padding = "24px";
  wrapper.style.position = "absolute";
  wrapper.style.top = "0";
  wrapper.style.left = "0";
  wrapper.style.opacity = "0";
  wrapper.style.pointerEvents = "none";
  wrapper.style.zIndex = "-1";
  wrapper.style.width = `${element.scrollWidth}px`;
  wrapper.appendChild(clone);
  document.body.appendChild(wrapper);

  // Force light-theme colours on the clone itself
  clone.style.backgroundColor = "#ffffff";
  clone.style.color = "#171717";

  // Allow the browser to finish layout before capturing
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  let imgData: string;
  try {
    imgData = await htmlToImage.toPng(wrapper, {
      pixelRatio: 1.5,
      backgroundColor: "#ffffff",
    });
  } catch (e) {
    throw new Error(
      `Image capture failed: ${e instanceof Error ? e.message : String(e)}`
    );
  } finally {
    document.body.removeChild(wrapper);
  }

  // A4 page dimensions in mm
  const pageWidth = 210;
  const pageHeight = 297;
  const margin = 10;
  const usableWidth = pageWidth - margin * 2;

  // Get image dimensions from the data URL
  const img = new Image();
  img.src = imgData;
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Failed to load captured image"));
  });

  const imgWidth = img.naturalWidth;
  const imgHeight = img.naturalHeight;
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
