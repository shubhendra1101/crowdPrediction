import "./globals.css";

export const metadata = { title: "CrowdSafe replay", description: "Zone density, forecast and alerts (replay)" };

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
