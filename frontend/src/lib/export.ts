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
 * Renders inside an off-screen iframe with light-theme overrides
 * so the PDF is white-background / dark-text.
 */
export async function exportAsPDF(
  element: HTMLElement,
  filename?: string
) {
  const safeName = filename || `answer-${Date.now()}.pdf`;
  const finalName = safeName.endsWith(".pdf") ? safeName : `${safeName}.pdf`;

  // Collect all CSS rules from the current page
  let allCss = "";
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      allCss +=
        Array.from(sheet.cssRules)
          .map((r) => r.cssText)
          .join("\n") + "\n";
    } catch {
      // Cross-origin stylesheet — skip
    }
  }

  // Build an off-screen iframe with light-theme overrides
  const iframe = document.createElement("iframe");
  iframe.style.cssText =
    "position:fixed;top:0;left:0;width:100%;height:100%;opacity:0;pointer-events:none;z-index:-1;border:none;";
  document.body.appendChild(iframe);

  const doc = iframe.contentDocument!;
  doc.open();
  doc.write(`
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          ${allCss}
          body {
            background: #ffffff !important;
            color: #171717 !important;
            padding: 24px;
            font-family: system-ui, -apple-system, sans-serif;
            line-height: 1.6;
          }
          * { color: #171717 !important; }
          a { color: #2563eb !important; text-decoration: underline !important; }
          pre, code { background-color: #f5f5f5 !important; color: #171717 !important; }
          blockquote { border-left-color: #d4d4d4 !important; }
        </style>
      </head>
      <body>${element.outerHTML}</body>
    </html>
  `);
  doc.close();

  // Allow browser to render
  await new Promise<void>((resolve) => {
    requestAnimationFrame(() => setTimeout(resolve, 100));
  });

  let imgData: string;
  try {
    imgData = await htmlToImage.toPng(doc.body, {
      pixelRatio: 1.5,
      backgroundColor: "#ffffff",
    });
  } catch (e) {
    throw new Error(
      `Image capture failed: ${e instanceof Error ? e.message : String(e)}`
    );
  } finally {
    document.body.removeChild(iframe);
  }

  // A4 page dimensions in mm
  const pageWidth = 210;
  const pageHeight = 297;
  const margin = 10;
  const usableWidth = pageWidth - margin * 2;

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

  pdf.addImage(imgData, "PNG", margin, margin, scaledWidth, scaledHeight);
  heightLeft -= pageHeight - margin * 2;

  while (heightLeft > 0) {
    position = heightLeft - scaledHeight + margin;
    pdf.addPage();
    pdf.addImage(imgData, "PNG", margin, position, scaledWidth, scaledHeight);
    heightLeft -= pageHeight - margin * 2;
  }

  pdf.save(finalName);
}
