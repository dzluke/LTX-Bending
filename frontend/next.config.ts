import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  allowedDevOrigins: ["tunnels-pod2.", "tunnels-pod2.:10081", "*.tunnels-pod2."],
  images: { unoptimized: true },
};

export default nextConfig;
