import { MessageSquare, Plus, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ThreadMeta {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
}

interface SidebarProps {
  threads: ThreadMeta[];
  activeThreadId: string | null;
  isOpen: boolean;
  onToggle: () => void;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const isToday =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (isToday) {
    return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

export function Sidebar({
  threads,
  activeThreadId,
  isOpen,
  onToggle,
  onSelect,
  onNewChat,
  onDelete,
}: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed md:relative z-50 h-full bg-neutral-900 border-r border-neutral-700 flex flex-col transition-all duration-200",
          isOpen ? "w-72 translate-x-0" : "w-0 -translate-x-full md:w-0 md:translate-x-0 overflow-hidden"
        )}
      >
        <div className="flex items-center justify-between p-3 border-b border-neutral-700">
          <h2 className="text-sm font-semibold text-neutral-300 flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            历史对话
          </h2>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-neutral-400 hover:text-white hover:bg-neutral-700"
              onClick={onNewChat}
              title="新建对话"
            >
              <Plus className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-neutral-400 hover:text-white hover:bg-neutral-700 md:hidden"
              onClick={onToggle}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {threads.length === 0 && (
            <div className="text-xs text-neutral-500 text-center py-8">
              暂无历史对话
            </div>
          )}
          {threads.map((t) => (
            <div
              key={t.id}
              className={cn(
                "group flex items-center gap-2 rounded-lg px-3 py-2 cursor-pointer transition-colors",
                activeThreadId === t.id
                  ? "bg-neutral-700 text-white"
                  : "text-neutral-300 hover:bg-neutral-800 hover:text-white"
              )}
              onClick={() => onSelect(t.id)}
            >
              <MessageSquare className="h-3.5 w-3.5 shrink-0 opacity-60" />
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate">{t.title}</div>
                <div className="text-[10px] opacity-50">{formatTime(t.updatedAt)}</div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 opacity-0 group-hover:opacity-100 text-neutral-400 hover:text-red-400 hover:bg-neutral-700 shrink-0"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(t.id);
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      </aside>

      {/* Toggle button (visible when sidebar is closed on desktop) */}
      {!isOpen && (
        <Button
          variant="outline"
          size="sm"
          className="fixed top-4 left-4 z-30 bg-neutral-800 border-neutral-600 text-neutral-200 hover:bg-neutral-700 hover:text-white hidden md:flex"
          onClick={onToggle}
        >
          <MessageSquare className="h-4 w-4 mr-1" />
          历史
        </Button>
      )}
    </>
  );
}
