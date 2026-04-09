import { config } from "dotenv";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
config({ path: resolve(__dirname, "../../../.env") });

/** Shared storage path — the NFS mount point on the host. */
export const SHARED_STORAGE = process.env.SHARED_STORAGE_PATH || "/mnt/tank";
