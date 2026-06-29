"use client";

import { useEffect, useState, useCallback } from "react";
import { toast } from "sonner";
import { listRegistries, createRegistry, deleteRegistry, updateRegistry, type SparkrunRegistry } from "@/lib/api";

const EMPTY = { name: "", url: "", subpath: "recipes" };

export function RegistriesSection() {
  const [rows, setRows] = useState<SparkrunRegistry[]>([]);
  const [form, setForm] = useState(EMPTY);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try { setRows(await listRegistries()); } catch (e) { toast.error((e as Error).message); }
  }, []);
  useEffect(() => { load(); }, [load]);

  async function add() {
    setBusy(true);
    try {
      await createRegistry(form);
      toast.success(`Added registry '${form.name}' — pushed to online nodes`);
      setForm(EMPTY);
      await load();
    } catch (e) { toast.error((e as Error).message); } finally { setBusy(false); }
  }

  async function remove(r: SparkrunRegistry) {
    if (!confirm(`Delete registry '${r.name}'? It will be removed from all nodes.`)) return;
    try { await deleteRegistry(r.id); toast.success(`Removed '${r.name}'`); await load(); }
    catch (e) { toast.error((e as Error).message); }
  }

  async function toggleVisible(r: SparkrunRegistry) {
    try {
      await updateRegistry(r.id, { visible: !r.visible });
      toast.success(`Registry '${r.name}' is now ${!r.visible ? "visible" : "hidden"}`);
      await load();
    } catch (e) { toast.error((e as Error).message); }
  }

  return (
    <section className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <h2 className="text-lg font-semibold mb-4">Sparkrun Registries</h2>
      <p className="text-sm text-gray-400 mb-4">
        Recipe registries cloned by sparkrun on every node. Changes are pushed to all online nodes immediately.
      </p>

      <div className="flex flex-wrap gap-2 mb-6">
        <input className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm" placeholder="name (a-z0-9-)"
          value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
        <input className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm flex-1 min-w-[16rem]" placeholder="git url"
          value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })} />
        <input className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm" placeholder="subpath"
          value={form.subpath} onChange={(e) => setForm({ ...form, subpath: e.target.value })} />
        <button disabled={busy || !form.name || !form.url} onClick={add}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 rounded px-3 py-1 text-sm transition-colors">
          Add
        </button>
      </div>

      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-500 text-left border-b border-gray-800">
            <th className="pb-2 font-medium">Name</th>
            <th className="pb-2 font-medium">URL</th>
            <th className="pb-2 font-medium">Subpath</th>
            <th className="pb-2 font-medium">Visible</th>
            <th className="pb-2 font-medium"></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b border-gray-800/50">
              <td className="py-2 font-mono">{r.name}</td>
              <td className="py-2 text-gray-400 truncate max-w-[20rem]">{r.url}</td>
              <td className="py-2 text-gray-400 font-mono">{r.subpath}</td>
              <td className="py-2">
                <button onClick={() => toggleVisible(r)} className={r.visible ? "text-green-400" : "text-gray-600"}>
                  {r.visible ? "visible" : "hidden"}
                </button>
              </td>
              <td className="py-2">
                <button onClick={() => remove(r)} className="text-red-500 hover:text-red-400 text-xs transition-colors">Delete</button>
              </td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr><td colSpan={5} className="py-4 text-center text-gray-600">No registries</td></tr>
          )}
        </tbody>
      </table>
    </section>
  );
}
