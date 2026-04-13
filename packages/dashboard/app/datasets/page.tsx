"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { apiFetch, apiUpload } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";

interface Dataset {
  id: string;
  name: string;
  description: string | null;
  format: string;
  source: string;
  path: string | null;
  huggingfaceId: string | null;
  size: number | null;
  sampleCount: number | null;
  createdAt: string;
}

type Tab = "upload" | "huggingface" | "path";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

const FORMAT_COLORS: Record<string, string> = {
  sharegpt: "bg-purple-900/50 text-purple-300",
  openai: "bg-blue-900/50 text-blue-300",
  instruct: "bg-amber-900/50 text-amber-300",
  qa: "bg-cyan-900/50 text-cyan-300",
  jsonl: "bg-gray-800 text-gray-300",
  other: "bg-gray-800 text-gray-400",
};

const SOURCE_COLORS: Record<string, string> = {
  upload: "bg-green-900/50 text-green-300",
  huggingface: "bg-yellow-900/50 text-yellow-300",
  path: "bg-gray-800 text-gray-300",
};

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("upload");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Upload form
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadDesc, setUploadDesc] = useState("");
  const [uploadFormat, setUploadFormat] = useState("auto");

  // HuggingFace form
  const [hfId, setHfId] = useState("");
  const [hfName, setHfName] = useState("");
  const [hfDesc, setHfDesc] = useState("");

  // Path form
  const [pathValue, setPathValue] = useState("");
  const [pathName, setPathName] = useState("");

  // Preview
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [previewRows, setPreviewRows] = useState<unknown[] | null>(null);
  const [previewMsg, setPreviewMsg] = useState<string | null>(null);

  const loadData = useCallback(() => {
    apiFetch<Dataset[]>("/api/datasets")
      .then(setDatasets)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleSSE = useCallback((event: SseEvent) => {
    if (event.type === "dataset:created") {
      setDatasets((prev) => [event.payload as unknown as Dataset, ...prev]);
    }
    if (event.type === "dataset:deleted") {
      const { id } = event.payload as { id: string };
      setDatasets((prev) => prev.filter((d) => d.id !== id));
      if (previewId === id) {
        setPreviewId(null);
        setPreviewRows(null);
      }
    }
  }, [previewId]);

  useSSE(handleSSE, loadData);

  function resetForms() {
    setUploadName("");
    setUploadDesc("");
    setUploadFormat("auto");
    setHfId("");
    setHfName("");
    setHfDesc("");
    setPathValue("");
    setPathName("");
    if (fileRef.current) fileRef.current.value = "";
    setError(null);
  }

  async function handleUpload() {
    const file = fileRef.current?.files?.[0];
    if (!file) return setError("Select a file");
    const name = uploadName || file.name.replace(/\.[^.]+$/, "");
    if (!name) return setError("Name is required");

    setSubmitting(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("name", name);
      if (uploadDesc) fd.append("description", uploadDesc);
      fd.append("format", uploadFormat);
      await apiUpload<Dataset>("/api/datasets", fd);
      resetForms();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleHfImport() {
    if (!hfId) return setError("HuggingFace dataset ID is required");
    const name = hfName || hfId.split("/").pop() || hfId;

    setSubmitting(true);
    setError(null);
    try {
      await apiFetch<Dataset>("/api/datasets", {
        method: "POST",
        body: JSON.stringify({
          name,
          description: hfDesc || null,
          source: "huggingface",
          huggingfaceId: hfId,
        }),
      });
      resetForms();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function handlePathRegister() {
    if (!pathValue) return setError("Path is required");
    const name = pathName || pathValue.split("/").pop() || pathValue;

    setSubmitting(true);
    setError(null);
    try {
      await apiFetch<Dataset>("/api/datasets", {
        method: "POST",
        body: JSON.stringify({ name, source: "path", path: pathValue }),
      });
      resetForms();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Register failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this dataset?")) return;
    await apiFetch(`/api/datasets/${id}`, { method: "DELETE" });
  }

  async function togglePreview(id: string) {
    if (previewId === id) {
      setPreviewId(null);
      setPreviewRows(null);
      setPreviewMsg(null);
      return;
    }
    setPreviewId(id);
    setPreviewRows(null);
    setPreviewMsg(null);
    try {
      const data = await apiFetch<{ preview: unknown[]; message?: string }>(
        `/api/datasets/${id}/preview`
      );
      setPreviewRows(data.preview);
      setPreviewMsg(data.message || null);
    } catch {
      setPreviewMsg("Failed to load preview");
    }
  }

  function onFileChange() {
    const file = fileRef.current?.files?.[0];
    if (file && !uploadName) {
      setUploadName(file.name.replace(/\.[^.]+$/, ""));
    }
  }

  const tabClass = (t: Tab) =>
    `px-4 py-2 text-sm font-medium rounded-t transition-colors ${
      tab === t
        ? "bg-gray-800 text-white border-b-2 border-green-400"
        : "text-gray-400 hover:text-gray-200"
    }`;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Datasets</h1>

      {/* Import Form */}
      <section className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="flex gap-1 px-4 pt-4">
          <button className={tabClass("upload")} onClick={() => setTab("upload")}>
            Upload File
          </button>
          <button className={tabClass("huggingface")} onClick={() => setTab("huggingface")}>
            HuggingFace
          </button>
          <button className={tabClass("path")} onClick={() => setTab("path")}>
            Existing Path
          </button>
        </div>

        <div className="p-4">
          {tab === "upload" && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">File (.jsonl, .json, .csv)</label>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".jsonl,.json,.csv"
                  onChange={onFileChange}
                  className="w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:bg-gray-800 file:text-gray-300 hover:file:bg-gray-700"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Name</label>
                  <input
                    type="text"
                    value={uploadName}
                    onChange={(e) => setUploadName(e.target.value)}
                    placeholder="Auto-filled from filename"
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Format</label>
                  <select
                    value={uploadFormat}
                    onChange={(e) => setUploadFormat(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                  >
                    <option value="auto">Auto-detect</option>
                    <option value="sharegpt">ShareGPT</option>
                    <option value="openai">OpenAI</option>
                    <option value="instruct">Instruct</option>
                    <option value="qa">QA</option>
                    <option value="jsonl">JSONL (raw)</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Description (optional)</label>
                <input
                  type="text"
                  value={uploadDesc}
                  onChange={(e) => setUploadDesc(e.target.value)}
                  placeholder="What is this dataset for?"
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                />
              </div>
              <button
                onClick={handleUpload}
                disabled={submitting}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 rounded text-sm font-medium transition-colors"
              >
                {submitting ? "Uploading..." : "Upload Dataset"}
              </button>
            </div>
          )}

          {tab === "huggingface" && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">HuggingFace Dataset ID</label>
                <input
                  type="text"
                  value={hfId}
                  onChange={(e) => setHfId(e.target.value)}
                  placeholder="b-mc2/sql-create-context"
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Name (optional)</label>
                  <input
                    type="text"
                    value={hfName}
                    onChange={(e) => setHfName(e.target.value)}
                    placeholder="Auto-filled from ID"
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Description (optional)</label>
                  <input
                    type="text"
                    value={hfDesc}
                    onChange={(e) => setHfDesc(e.target.value)}
                    placeholder="What is this dataset for?"
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                  />
                </div>
              </div>
              <button
                onClick={handleHfImport}
                disabled={submitting}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 rounded text-sm font-medium transition-colors"
              >
                {submitting ? "Importing..." : "Import from HuggingFace"}
              </button>
            </div>
          )}

          {tab === "path" && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">File Path (on shared storage)</label>
                <input
                  type="text"
                  value={pathValue}
                  onChange={(e) => setPathValue(e.target.value)}
                  placeholder="/mnt/tank/data/my-dataset.jsonl"
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Name (optional)</label>
                <input
                  type="text"
                  value={pathName}
                  onChange={(e) => setPathName(e.target.value)}
                  placeholder="Auto-filled from filename"
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                />
              </div>
              <button
                onClick={handlePathRegister}
                disabled={submitting}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 rounded text-sm font-medium transition-colors"
              >
                {submitting ? "Registering..." : "Register Path"}
              </button>
            </div>
          )}

          {error && (
            <p className="mt-3 text-sm text-red-400">{error}</p>
          )}
        </div>
      </section>

      {/* Dataset List */}
      <section className="bg-gray-900 rounded-lg border border-gray-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-left border-b border-gray-800">
              <th className="px-4 py-3 font-medium">Name</th>
              <th className="px-4 py-3 font-medium">Format</th>
              <th className="px-4 py-3 font-medium">Source</th>
              <th className="px-4 py-3 font-medium">Size</th>
              <th className="px-4 py-3 font-medium">Samples</th>
              <th className="px-4 py-3 font-medium">Created</th>
              <th className="px-4 py-3 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-600">
                  Loading...
                </td>
              </tr>
            )}
            {!loading && datasets.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-600">
                  No datasets yet. Upload one above.
                </td>
              </tr>
            )}
            {datasets.map((d) => (
              <tr key={d.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="px-4 py-3">
                  <button
                    onClick={() => togglePreview(d.id)}
                    className="text-left hover:text-green-400 transition-colors"
                  >
                    <span className="font-medium">{d.name}</span>
                    {d.description && (
                      <span className="block text-xs text-gray-500">{d.description}</span>
                    )}
                  </button>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${FORMAT_COLORS[d.format] || FORMAT_COLORS.other}`}>
                    {d.format}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${SOURCE_COLORS[d.source] || SOURCE_COLORS.path}`}>
                    {d.source === "huggingface" ? "HF" : d.source}
                  </span>
                  {d.huggingfaceId && (
                    <span className="ml-1 text-xs text-gray-500">{d.huggingfaceId}</span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-400">
                  {d.size ? formatBytes(d.size) : "-"}
                </td>
                <td className="px-4 py-3 text-gray-400">
                  {d.sampleCount?.toLocaleString() || "-"}
                </td>
                <td className="px-4 py-3 text-gray-400">
                  {new Date(d.createdAt).toLocaleDateString()}
                </td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => handleDelete(d.id)}
                    className="text-red-500 hover:text-red-400 text-xs transition-colors"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Preview Panel */}
        {previewId && (
          <div className="border-t border-gray-800 p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-300">Preview</h3>
              <button
                onClick={() => { setPreviewId(null); setPreviewRows(null); }}
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                Close
              </button>
            </div>
            {previewMsg && (
              <p className="text-sm text-gray-500 italic">{previewMsg}</p>
            )}
            {!previewRows && !previewMsg && (
              <p className="text-sm text-gray-500">Loading preview...</p>
            )}
            {previewRows && previewRows.length > 0 && (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {previewRows.map((row, i) => (
                  <pre
                    key={i}
                    className="bg-gray-950 rounded p-3 text-xs text-gray-300 overflow-x-auto"
                  >
                    {JSON.stringify(row, null, 2)}
                  </pre>
                ))}
              </div>
            )}
            {previewRows && previewRows.length === 0 && !previewMsg && (
              <p className="text-sm text-gray-500">No data rows found</p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
