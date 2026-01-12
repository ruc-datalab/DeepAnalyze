"use client";

import type React from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import Editor from "@monaco-editor/react";
import { useState, useRef, useEffect, useCallback, memo } from "react";
import { configureMonaco } from "@/lib/monaco-config";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Card } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { API_URLS, API_CONFIG } from "@/lib/config";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import {
  Send,
  Sparkles,
  User,
  Paperclip,
  X,
  FileText,
  ImageIcon,
  ChevronDown,
  ChevronRight,
  Trash2,
  Download,
  Play,
  Save,
  FolderOpen,
  RefreshCw,
  Moon,
  Sun,
  Eraser,
  Copy,
  Check,
  Edit,
  Upload,
  Square,
  Code2,
} from "lucide-react";
import { Tree, NodeApi } from "react-arborist";
import { useToast } from "@/hooks/use-toast";
import { FileIcon, defaultStyles } from "react-file-icon";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
  attachments?: FileAttachment[];
  localOnly?: boolean;
}

interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
}

interface WorkspaceFile {
  name: string;
  size: number;
  extension: string;
  icon: string;
  download_url: string;
  preview_url?: string;
}

type WorkspaceNode = {
  name: string;
  path: string; // relative path
  is_dir: boolean;
  size?: number;
  extension?: string;
  icon?: string;
  download_url?: string;
  children?: WorkspaceNode[];
  is_generated?: boolean; // 标识是否为代码生成的文件或文件夹
};

interface AnalysisSection {
  type: "Analyze" | "Understand" | "Code" | "Execute" | "Answer";
  content: string;
  icon: string;
  color: string;
}

type CodeBlockViewProps = {
  language: string;
  code: string;
  showHeader?: boolean;
  isDarkMode: boolean;
  onEdit: (code: string) => void;
};

const CodeBlockView = memo(function CodeBlockView({
  language,
  code,
  showHeader = false,
  isDarkMode,
  onEdit,
}: CodeBlockViewProps) {
  const { toast } = useToast();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code.trim());
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 1500);
      toast({ description: "已复制代码" });
    } catch {
      toast({ description: "复制失败", variant: "destructive" });
    }
  };

  const isLargeCode = code.length > 8000;

  return (
    <div className="code-block my-3 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      {showHeader && (
        <div className="flex items-center justify-between bg-gray-50 dark:bg-gray-800 px-3 py-2 text-xs">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="h-5 w-5 p-0 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              {isCollapsed ? (
                <ChevronRight className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </Button>
            <span className="text-gray-600 dark:text-gray-300">Code</span>
            <span className="text-gray-500 font-mono">{language || "text"}</span>
            {isLargeCode && (
              <span className="text-[10px] text-gray-400">
                （代码较长，已关闭高亮）
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="h-5 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              {isCopied ? (
                <Check className="h-3 w-3" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onEdit(code.trim())}
              className="h-5 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <Edit className="h-3 w-3" />
            </Button>
          </div>
        </div>
      )}
      {!showHeader || !isCollapsed ? (
        isLargeCode ? (
          <pre className="m-0 p-3 text-xs overflow-x-auto whitespace-pre-wrap font-mono bg-transparent">
            {code.trim()}
          </pre>
        ) : (
          <SyntaxHighlighter
            language={language || "text"}
            style={isDarkMode ? oneDark : oneLight}
            customStyle={{
              margin: 0,
              background: "transparent",
              overflowX: "hidden",
              whiteSpace: "pre-wrap",
            }}
            codeTagProps={{
              style: {
                fontFamily: "var(--font-mono)",
                fontSize: "0.875rem",
                whiteSpace: "pre-wrap",
              },
            }}
          >
            {code.trim()}
          </SyntaxHighlighter>
        )
      ) : null}
    </div>
  );
});

type ChatMessageItemProps = {
  message: Message;
  messageIndex: number;
  isStreaming: boolean;
  renderAssistant: (content: string, messageIndex?: number) => React.ReactNode;
  renderAssistantStreaming: (content: string, messageIndex?: number) => React.ReactNode;
};

const ChatMessageItem = memo(
  function ChatMessageItem({
    message,
    messageIndex,
    isStreaming,
    renderAssistant,
    renderAssistantStreaming,
  }: ChatMessageItemProps) {
    return (
      <div className="space-y-2">
        {message.sender === "user" ? (
          <div className="flex items-start justify-end gap-2">
            <div className="max-w-[80%] bg-black text-white dark:bg-white dark:text-black rounded-lg px-4 py-3 message-bubble message-appear">
              <div className="text-sm break-words whitespace-pre-wrap">
                {message.content}
              </div>
            </div>
            <Avatar>
              <AvatarImage src="/placeholder-user.jpg" alt="User" />
              <AvatarFallback className="text-[10px]">U</AvatarFallback>
            </Avatar>
          </div>
        ) : (
          <div className="flex items-start gap-2 min-w-0">
            <Avatar>
              <AvatarImage src="/placeholder-logo.png" alt="AI Assistant" />
              <AvatarFallback className="text-[10px]">
                <Sparkles className="h-3 w-3" />
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1 message-appear">
              <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                Assistant
              </div>
              <div className="space-y-4 min-w-0">
                {isStreaming ? (
                  renderAssistantStreaming(message.content, messageIndex)
                ) : (
                  renderAssistant(message.content, messageIndex)
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  },
  (prev, next) => {
    return (
      prev.message === next.message &&
      prev.messageIndex === next.messageIndex &&
      prev.isStreaming === next.isStreaming &&
      prev.renderAssistant === next.renderAssistant &&
      prev.renderAssistantStreaming === next.renderAssistantStreaming
    );
  }
);

type StructuredSectionType =
  | "Analyze"
  | "Understand"
  | "Code"
  | "Execute"
  | "Answer"
  | "File";

const StreamingMarkdownBlock = memo(
  function StreamingMarkdownBlock({
    content,
    renderMarkdownContent,
    className,
  }: {
    content: string;
    renderMarkdownContent: (content: string) => React.ReactNode;
    className?: string;
  }) {
    if (!content.trim()) return null;
    return <div className={className}>{renderMarkdownContent(content)}</div>;
  },
  (prev, next) =>
    prev.content === next.content &&
    prev.renderMarkdownContent === next.renderMarkdownContent &&
    prev.className === next.className
);

const StreamingSectionBody = memo(
  function StreamingSectionBody({
    type,
    content,
    isComplete,
    renderSectionContent,
  }: {
    type: StructuredSectionType;
    content: string;
    isComplete: boolean;
    renderSectionContent: (content: string) => React.ReactNode;
  }) {
    if (!content.trim()) return null;
    if (!isComplete) {
      if (type === "Code" || type === "Execute") {
        return (
          <pre className="m-0 text-xs overflow-x-auto whitespace-pre-wrap font-mono">
            {content}
          </pre>
        );
      }
      return (
        <div className="text-sm break-words whitespace-pre-wrap">{content}</div>
      );
    }
    return <div className="markdown-content">{renderSectionContent(content)}</div>;
  },
  (prev, next) =>
    prev.type === next.type &&
    prev.content === next.content &&
    prev.isComplete === next.isComplete &&
    prev.renderSectionContent === next.renderSectionContent
);

export function ThreePanelInterface() {
  const { toast } = useToast();
  const [isDarkMode, setIsDarkMode] = useState(false); // 服务端默认 false
  const [mounted, setMounted] = useState(false);
  const [editorHeight, setEditorHeight] = useState(60); // 编辑器高度百分比
  const [collapsedSections, setCollapsedSections] = useState<
    Record<string, boolean>
  >({});
  const [autoCollapseEnabled, setAutoCollapseEnabled] = useState(true);
  const [manualLocks, setManualLocks] = useState<Record<string, boolean>>({});

  // Session ID：用于区分不同浏览器用户（无需登录）
  const [sessionId, setSessionId] = useState<string>("");

  // 步骤导航相关状态
  const [activeSection, setActiveSection] = useState<string>("");
  const stepNavigatorRef = useRef<HTMLDivElement>(null);
  const activeStepRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // 组件挂载后从 localStorage 读取主题
  useEffect(() => {
    setMounted(true);
    if (typeof window !== "undefined") {
      // 配置 Monaco Editor
      configureMonaco();

      // 初始化或获取 sessionId
      let sid = localStorage.getItem("sessionId");
      if (!sid) {
        sid = `session_${Date.now()}_${Math.random()
          .toString(36)
          .substr(2, 9)}`;
        localStorage.setItem("sessionId", sid);
      }
      setSessionId(sid);

      const savedTheme = localStorage.getItem("theme");
      const shouldBeDark = savedTheme === "dark";
      setIsDarkMode(shouldBeDark);
      updateThemeClass(shouldBeDark);
      const savedAuto = localStorage.getItem("autoCollapseEnabled");
      if (savedAuto !== null) {
        setAutoCollapseEnabled(savedAuto !== "false");
      }
    }
  }, []);

  // 按 session 维度持久化/恢复 折叠状态 与 手动锁
  useEffect(() => {
    if (!sessionId) return;
    try {
      const cs = localStorage.getItem(`collapsedSections:${sessionId}`);
      if (cs) setCollapsedSections(JSON.parse(cs));
      const ml = localStorage.getItem(`manualLocks:${sessionId}`);
      if (ml) setManualLocks(JSON.parse(ml));
    } catch { }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    try {
      localStorage.setItem(
        `collapsedSections:${sessionId}`,
        JSON.stringify(collapsedSections)
      );
      localStorage.setItem(
        `manualLocks:${sessionId}`,
        JSON.stringify(manualLocks)
      );
    } catch { }
  }, [sessionId, collapsedSections, manualLocks]);

  // 当 activeSection 变化时自动滚动到对应步骤
  useEffect(() => {
    if (activeSection && stepNavigatorRef.current) {
      const activeStepElement = activeStepRefs.current.get(activeSection);
      if (activeStepElement) {
        const container = stepNavigatorRef.current;
        const stepRect = activeStepElement.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();

        // 计算需要滚动的距离
        const scrollLeft =
          activeStepElement.offsetLeft -
          containerRect.width / 2 +
          stepRect.width / 2;

        // 平滑滚动到目标位置
        container.scrollTo({
          left: scrollLeft,
          behavior: "smooth",
        });
      }
    }
  }, [activeSection]);

  // 更新主题 class
  const updateThemeClass = (isDark: boolean) => {
    if (typeof document !== "undefined") {
      if (isDark) {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    }
  };

  // 获取某条消息之前最近的用户问题内容
  const getPrevUserQuestionText = (index: number): string => {
    for (let i = index - 1; i >= 0; i--) {
      const m = messages[i];
      if (m && m.sender === "user") return m.content || "";
    }
    return "";
  };

  const buildReportFilename = (question: string) => {
    const clean = (question || "").replace(/\s+/g, " ").trim();
    let tokens = clean.split(/\s+/).filter(Boolean);
    let base = "";
    if (tokens.length <= 1) {
      // 中文/无空格：直接取前 5 个字符，不再用下划线
      base = clean.replace(/\s+/g, "").slice(0, 5);
    } else {
      // 英文/有空格：取前 5 个词，用下划线连接
      base = tokens
        .slice(0, 5)
        .map((t) => t.replace(/[\\/:*?"<>|]/g, ""))
        .filter(Boolean)
        .join("_");
    }
    base = base.slice(0, 120);
    return `Report_${base || "Untitled"}.pdf`;
  };

  const exportReportBackend = async () => {
    try {
      const payloadMessages = messages
        .filter((m) => !m.localOnly)
        .map((msg) => ({
          role: msg.sender === "user" ? "user" : "assistant",
          content: msg.content,
        }));
      const title = getPrevUserQuestionText(messages.length);
      const res = await fetch(API_URLS.EXPORT_REPORT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: payloadMessages,
          title,
          session_id: sessionId,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const md = data?.md;
      toast({ description: `已提交并生成: ${md}` });
      await loadWorkspaceFiles();
      await loadWorkspaceTree?.();
    } catch (e) {
      console.error("backend export error", e);
      toast({ description: "导出失败", variant: "destructive" });
    }
  };
  const exportReportBackendRef = useRef(exportReportBackend);
  useEffect(() => {
    exportReportBackendRef.current = exportReportBackend;
  }, [exportReportBackend]);

  // 切换主题
  const toggleTheme = () => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);
    updateThemeClass(newDarkMode);

    // 保存到 localStorage
    if (typeof window !== "undefined") {
      localStorage.setItem("theme", newDarkMode ? "dark" : "light");
    }
  };

  // 处理拖动调整大小
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = editorHeight;

    const handleMouseMove = (e: MouseEvent) => {
      const container = document.querySelector(".editor-container");
      if (!container) return;

      const containerRect = container.getBoundingClientRect();
      const deltaY = e.clientY - startY;
      const containerHeight = containerRect.height;
      const deltaPercent = (deltaY / containerHeight) * 100;

      const newHeight = Math.min(Math.max(startHeight + deltaPercent, 20), 80);
      setEditorHeight(newHeight);
    };

    const handleMouseUp = () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome-1",
      content: "Hello! I'm DeepAnalyze-8B, your autonomous data science assistant. Upload your data and let's explore it together!",
      sender: "ai",
      timestamp: new Date(),
      localOnly: true,
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [attachments, setAttachments] = useState<FileAttachment[]>([]);
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFile[]>([]);
  const [workspaceTree, setWorkspaceTree] = useState<WorkspaceNode | null>(
    null
  );
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const treeContainerRef = useRef<HTMLDivElement>(null);
  const [treeSize, setTreeSize] = useState<{ w: number; h: number }>({
    w: 0,
    h: 0,
  });
  const [selectedCodeSection, setSelectedCodeSection] = useState<string>("");
  const [codeEditorContent, setCodeEditorContent] = useState("");
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  const [isExecutingCode, setIsExecutingCode] = useState(false);
  const [codeExecutionResult, setCodeExecutionResult] = useState("");

  // 预览弹窗状态
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [previewTitle, setPreviewTitle] = useState<string>("");
  const [previewContent, setPreviewContent] = useState<string>("");
  const [previewType, setPreviewType] = useState<
    "text" | "image" | "pdf" | "binary"
  >("text");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewDownloadUrl, setPreviewDownloadUrl] = useState<string>("");
  const previewScrollRef = useRef<HTMLDivElement>(null);
  const [deleteConfirmPath, setDeleteConfirmPath] = useState<string | null>(
    null
  );
  const [deleteIsDir, setDeleteIsDir] = useState<boolean>(false);
  const fileRefreshTimerRef = useRef<number | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const singleClickTimerRef = useRef<number | null>(null);
  const [contextPos, setContextPos] = useState<{ x: number; y: number } | null>(
    null
  );
  const [contextTarget, setContextTarget] = useState<WorkspaceNode | null>(
    null
  );
  const [dragOverPath, setDragOverPath] = useState<string | null>(null);
  const [dropActive, setDropActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string>("");

  const lastScrollTimeRef = useRef(0);
  const scrollRafRef = useRef<number | null>(null);
  const stickToBottomRef = useRef(true);
  // const aiUpdateTimerRef = useRef<number | null>(null); // Removed in favor of RAF
  const aiPendingContentRef = useRef<string>("");
  const aiDisplayedContentRef = useRef<string>("");
  const streamRafRef = useRef<number | null>(null);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(
    null
  );
  // const [clearChatOpen, setClearChatOpen] = useState(false); // Removed redundant state

  // 节流滚动到底部

  const scrollToBottom = useCallback((force: boolean = false) => {
    const now = Date.now();
    const timeSinceLastScroll = now - lastScrollTimeRef.current;

    // 节流：默认 100ms，强制模式下忽略
    if (!force && timeSinceLastScroll < 100) {
      return;
    }

    if (scrollRafRef.current) {
      cancelAnimationFrame(scrollRafRef.current);
    }

    scrollRafRef.current = requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        const container = messagesContainerRef.current;
        // 使用 behavior: auto (默认) 以确保瞬间跳转，避免 smooth 带来的滞后叠加
        container.scrollTop = container.scrollHeight;
        stickToBottomRef.current = true;
        lastScrollTimeRef.current = Date.now();
      }
      scrollRafRef.current = null;
    });
  }, []);

  // 输入完成后平滑滚动到底部（避免流式期间 setInterval 导致频繁布局计算）
  useEffect(() => {
    if (isTyping) return;
    if (!stickToBottomRef.current) return;
    setTimeout(() => {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTo({
          top: messagesContainerRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
    }, 100);
  }, [isTyping]);

  // 监听消息变化
  useEffect(() => {
    if (stickToBottomRef.current) {
      // 流式输出时(streamingMessageId存在)强制滚动，消除滞后
      scrollToBottom(!!streamingMessageId);
    }
  }, [messages, scrollToBottom, streamingMessageId]);

  // 聊天消息本地缓存：加载与保存
  const CHAT_STORAGE_KEY = "chat_messages_v1";
  const [chatLoaded, setChatLoaded] = useState(false);

  // 挂载后再次从本地覆盖加载，避免 SSR 初始状态覆盖缓存
  useEffect(() => {
    try {
      if (typeof window === "undefined") return;
      const raw = localStorage.getItem(CHAT_STORAGE_KEY);
      if (raw) {
        const arr = JSON.parse(raw) as any[];
        if (Array.isArray(arr) && arr.length) {
          const restored = arr.map((m) => ({
            ...m,
            timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
          })) as Message[];
          setMessages(restored);
        }
      }
    } catch (e) {
      console.warn("post-mount load chat cache failed", e);
    }
    setChatLoaded(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 消息本地缓存：流式生成时节流保存，避免每个 chunk 都写 localStorage 导致卡顿
  const saveChatTimerRef = useRef<number | null>(null);
  useEffect(() => {
    try {
      if (!chatLoaded) return; // 避免首屏用欢迎消息覆盖已有缓存
      if (typeof window === "undefined") return;

      if (saveChatTimerRef.current) {
        window.clearTimeout(saveChatTimerRef.current);
        saveChatTimerRef.current = null;
      }

      const delay = isTyping ? 1500 : 200;
      saveChatTimerRef.current = window.setTimeout(() => {
        try {
          const data = JSON.stringify(
            messages.map((m) => ({
              ...m,
              timestamp: (m.timestamp instanceof Date
                ? m.timestamp
                : new Date(m.timestamp as any)
              ).toISOString(),
            }))
          );
          localStorage.setItem(CHAT_STORAGE_KEY, data);
        } catch (e) {
          console.warn("save chat cache failed", e);
        } finally {
          saveChatTimerRef.current = null;
        }
      }, delay);
    } catch (e) {
      console.warn("save chat cache failed", e);
    }
  }, [messages, chatLoaded, isTyping]);

  // 一键清空聊天：保留欢迎消息（仅本地显示）
  const clearChat = () => {
    if (isTyping) {
      toast({ description: "执行中，暂时无法清空", variant: "destructive" });
      return;
    }
    const welcome: Message = {
      id: `welcome-${Date.now()}`,
      content: "Hello! I'm DeepAnalyze-8B, your autonomous data science assistant. Upload your data and let's explore it together!",
      sender: "ai",
      timestamp: new Date(),
      localOnly: true,
    };
    setMessages([welcome]);
    try {
      if (typeof window !== "undefined") {
        localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify([welcome]));
      }
    } catch { }
    toast({ description: "已清空聊天" });
  };

  useEffect(() => {
    if (sessionId) {
      loadWorkspaceFiles();
      loadWorkspaceTree();
    }
  }, [sessionId]);

  useEffect(() => {
    const id = setInterval(() => {
      // 智能轮询：仅在页面可见且未上传时轮询
      const isVisible =
        typeof document !== "undefined" && document.visibilityState === "visible";
      if (!isUploading && isVisible) {
        loadWorkspaceTree();
        loadWorkspaceFiles();
      }
    }, 4000);
    return () => clearInterval(id);
  }, [isUploading]);

  useEffect(() => {
    const el = treeContainerRef.current;
    if (!el) return;
    const ro = new (window as any).ResizeObserver((entries: any) => {
      for (const entry of entries) {
        const cr = entry.contentRect as DOMRectReadOnly;
        setTreeSize({
          w: Math.max(0, Math.floor(cr.width)),
          h: Math.max(0, Math.floor(cr.height)),
        });
      }
    });
    ro.observe(el);
    const rect = el.getBoundingClientRect();
    setTreeSize({
      w: Math.max(0, Math.floor(rect.width)),
      h: Math.max(0, Math.floor(rect.height)),
    });
    return () => ro.disconnect();
  }, []);

  const loadWorkspaceFiles = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(
        `${API_URLS.WORKSPACE_FILES}?session_id=${sessionId}`
      );
      if (response.ok) {
        const data = await response.json();
        setWorkspaceFiles(data.files);
      }
    } catch (error) {
      console.error("Failed to load workspace files:", error);
    }
  };

  const loadWorkspaceTree = async () => {
    if (!sessionId) return;
    try {
      const res = await fetch(
        `${API_URLS.WORKSPACE_TREE}?session_id=${sessionId}`
      );
      if (res.ok) {
        const data = await res.json();
        // 标记 generated 文件夹及其内容
        const markGenerated = (
          node: WorkspaceNode,
          parentIsGenerated = false
        ) => {
          const isGenerated =
            parentIsGenerated ||
            node.name === "generated" ||
            node.path.startsWith("generated/") ||
            node.path.startsWith("generated");
          node.is_generated = isGenerated;
          if (node.children) {
            node.children.forEach((child) => markGenerated(child, isGenerated));
          }
        };
        if (data) {
          markGenerated(data);
        }
        setWorkspaceTree(data);
        // 默认展开根与第一层，包括 generated 文件夹
        const init: Record<string, boolean> = { "": true };
        if (data?.children) {
          data.children.forEach((c: WorkspaceNode) => {
            if (c.is_dir) init[c.path] = true;
          });
        }
        setExpanded(init);
      }
    } catch (e) {
      console.error("load tree error", e);
    }
  };

  const toggleExpand = (p: string) =>
    setExpanded((prev) => ({ ...prev, [p]: !prev[p] }));

  const deleteFile = async (p: string) => {
    try {
      const url = `${API_URLS.WORKSPACE_DELETE_FILE}?path=${encodeURIComponent(
        p
      )}&session_id=${encodeURIComponent(sessionId)}`;
      const res = await fetch(url, { method: "DELETE" });
      if (res.ok) {
        await loadWorkspaceTree();
        await loadWorkspaceFiles();
      }
    } catch (e) {
      console.error("delete file error", e);
    }
  };

  const deleteDir = async (p: string) => {
    try {
      const url = `${API_URLS.WORKSPACE_DELETE_DIR}?path=${encodeURIComponent(
        p
      )}&recursive=true&session_id=${encodeURIComponent(sessionId)}`;
      const res = await fetch(url, { method: "DELETE" });
      if (res.ok) {
        await loadWorkspaceTree();
        await loadWorkspaceFiles();
      }
    } catch (e) {
      console.error("delete dir error", e);
    }
  };

  // 移动：将工作区内的文件/文件夹移动到指定目录（空字符串表示根目录）
  const moveToDir = async (srcPath: string, dstDir: string) => {
    try {
      const url = `${API_CONFIG.BACKEND_BASE_URL
        }/workspace/move?src=${encodeURIComponent(
          srcPath
        )}&dst_dir=${encodeURIComponent(dstDir)}&session_id=${encodeURIComponent(
          sessionId
        )}`;
      const res = await fetch(url, { method: "POST" });
      if (res.ok) {
        await loadWorkspaceTree();
        await loadWorkspaceFiles();
      }
    } catch (e) {
      console.error("move to dir error", e);
    }
  };

  const uploadToDir = async (dirPath: string, files: FileList | File[]) => {
    try {
      setIsUploading(true);
      const form = new FormData();
      const arr: File[] = Array.from(files as File[]);
      arr.forEach((f) => form.append("files", f));
      const url = `${API_URLS.WORKSPACE_UPLOAD_TO}?dir=${encodeURIComponent(
        dirPath || ""
      )}&session_id=${encodeURIComponent(sessionId)}`;
      await fetch(url, { method: "POST", body: form });
      await loadWorkspaceTree();
      await loadWorkspaceFiles();
      setUploadMsg(`上传成功 ${arr.length} 个文件`);
      setTimeout(() => setUploadMsg(""), 2000);
    } catch (e) {
      console.error("upload to dir error", e);
      setUploadMsg("上传失败");
      setTimeout(() => setUploadMsg(""), 2500);
    }
    setIsUploading(false);
  };

  const openNode = async (node: WorkspaceNode) => {
    if (node.is_dir) return;
    const ext = (node.extension || "").replace(/^\./, "").toLowerCase();
    // 修正 URL，确保包含 generated 路径
    const correctedUrl = ensureGeneratedInUrl(node.download_url || "");
    const mapped: WorkspaceFile = {
      name: node.name,
      size: node.size || 0,
      extension: ext,
      icon: node.icon || "",
      download_url: correctedUrl,
      preview_url: correctedUrl,
    };
    openPreview(mapped);
  };

  const onContextMenu = (e: React.MouseEvent, node: WorkspaceNode) => {
    e.preventDefault();
    setContextTarget(node);
    setContextPos({ x: e.clientX, y: e.clientY });
  };

  const closeContext = () => {
    setContextPos(null);
    setContextTarget(null);
  };

  // 将后端树转换为 Arborist 数据
  type ArborNode = {
    id: string;
    name: string;
    isDir: boolean;
    icon?: string;
    download_url?: string;
    extension?: string;
    size?: number;
    children?: ArborNode[];
    isGenerated?: boolean; // 标识是否为代码生成的文件
  };

  const toArbor = (node: WorkspaceNode): ArborNode => ({
    id: node.path || "",
    name: node.name || "workspace",
    isDir: node.is_dir,
    icon: node.icon,
    download_url: node.download_url,
    extension: node.extension,
    size: node.size,
    isGenerated: node.is_generated,
    children: node.children?.map(toArbor),
  });

  const getExt = (name?: string, ext?: string) => {
    const fromExt = (ext || "").replace(/^\./, "").toLowerCase();
    if (fromExt) return fromExt;
    if (!name) return "txt";
    const p = name.lastIndexOf(".");
    return p > -1 ? name.slice(p + 1).toLowerCase() : "txt";
  };

  const Row = ({
    node,
    style,
    dragHandle,
  }: {
    node: NodeApi<ArborNode>;
    style: React.CSSProperties;
    dragHandle?: (el: HTMLDivElement | null) => void;
  }) => {
    const data = node.data;
    const isDir = data.isDir;
    const isGenerated = data.isGenerated || false;
    const isGeneratedFolder = isDir && data.name === "generated";
    const ext = getExt(data.name, data.extension);

    return (
      <div style={style}>
        {/* Generated 分组标题 + 删除按钮（不遮挡、不受折叠影响） */}
        {isGeneratedFolder && (
          <div className="mt-2 mb-1 px-2 flex items-center justify-between select-none">
            <div className="flex items-center gap-2 text-[11px] text-purple-600 dark:text-purple-400">
              <span className="h-px w-4 bg-purple-200 dark:bg-purple-800" />
              <span className="font-medium">代码生成文件</span>
            </div>
            <button
              className="text-red-600 hover:text-red-700 p-1 rounded hover:bg-red-50 dark:hover:bg-red-950/20"
              aria-label="删除生成文件夹"
              title="删除生成文件夹"
              onClick={(e) => {
                e.stopPropagation();
                setDeleteIsDir(true);
                setDeleteConfirmPath(data.id);
              }}
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
        <div
          className={`flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-900 rounded px-2 py-1 ${isGenerated ? "bg-purple-50 dark:bg-purple-950/20" : ""
            }`}
          onClick={(e) => {
            if (isDir) {
              node.toggle();
              return;
            }
            if (singleClickTimerRef.current) {
              window.clearTimeout(singleClickTimerRef.current);
              singleClickTimerRef.current = null;
            }
            // 延迟触发预览，若短时间内发生双击会被取消
            singleClickTimerRef.current = window.setTimeout(() => {
              openNode({
                name: data.name,
                path: data.id,
                is_dir: false,
                download_url: data.download_url,
                extension: data.extension,
                size: data.size,
                icon: data.icon,
              } as any);
              singleClickTimerRef.current = null;
            }, 180);
          }}
          onDoubleClick={(e) => {
            if (isDir) return;
            e.stopPropagation();
            if (singleClickTimerRef.current) {
              window.clearTimeout(singleClickTimerRef.current);
              singleClickTimerRef.current = null;
            }
            if (data.download_url) {
              downloadFileByUrl(data.name, data.download_url);
            }
          }}
          onContextMenu={(e) =>
            onContextMenu(
              e as any,
              {
                name: data.name,
                path: data.id,
                is_dir: isDir,
                download_url: data.download_url,
                extension: data.extension,
                size: data.size,
                icon: data.icon,
              } as any
            )
          }
          onDragOver={(e) => {
            if (isDir) {
              e.preventDefault();
              e.dataTransfer.dropEffect = (e.dataTransfer.types || []).includes(
                "text/x-workspace-path"
              )
                ? "move"
                : "copy";
            }
          }}
          onDragEnter={(e) => {
            if (isDir) setDragOverPath(data.id);
          }}
          onDragLeave={(e) => {
            if (isDir) setDragOverPath(null);
          }}
          onDrop={(e) => {
            if (!isDir) return;
            e.preventDefault();
            uploadToDir(data.id, e.dataTransfer.files || []);
            setDragOverPath(null);
          }}
        >
          <div
            className="flex items-center gap-2 text-sm"
            ref={dragHandle}
            draggable={!isDir}
            onDragStart={(e) => {
              if (isDir) return;
              // 将工作区内路径放入自定义 MIME，供目标目录 onDrop 读取
              e.dataTransfer.setData("text/x-workspace-path", data.id);
              // 提示为移动操作
              e.dataTransfer.effectAllowed = "move";
            }}
          >
            {isDir ? (
              <>
                <span
                  className={
                    isGenerated
                      ? "text-purple-600 dark:text-purple-400"
                      : "text-gray-500"
                  }
                >
                  {node.isOpen ? "▾" : "▸"}
                </span>
                {isGenerated ? (
                  <Code2 className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" />
                ) : (
                  <FolderOpen className="h-3.5 w-3.5 text-gray-500" />
                )}
              </>
            ) : (
              <div style={{ width: 16, height: 16 }}>
                {/* 动态扩展样式，fallback 到 txt */}
                {/* @ts-ignore */}
                <FileIcon
                  extension={ext}
                  {...((defaultStyles as any)[ext] ||
                    (defaultStyles as any).txt)}
                />
              </div>
            )}
            <span
              className={`truncate ${isGenerated
                ? "text-purple-700 dark:text-purple-300 font-medium"
                : ""
                }`}
            >
              {data.name}
            </span>
            {typeof data.size === "number" && !isDir && (
              <span className="text-[10px] text-gray-400 ml-2 shrink-0">
                {formatFileSize(data.size)}
              </span>
            )}
            {isGenerated && !isDir && (
              <Sparkles className="h-3 w-3 text-purple-500 ml-1 shrink-0" />
            )}
          </div>
          {/* 行尾不再展示下载/删除按钮。双击/点击行为保持不变；右键菜单提供下载/删除。*/}
        </div>
      </div>
    );
  };

  const renderTree = (node: WorkspaceNode, depth = 0) => {
    const isDir = node.is_dir;
    const isGenerated = node.is_generated || false;
    const isGeneratedFolder = isDir && node.name === "generated" && depth === 1;
    const pad = { paddingLeft: `${8 + depth * 14}px` } as React.CSSProperties;

    return (
      <div key={node.path || "root"}>
        {/* Generated 文件夹上方添加分隔线 */}
        {isGeneratedFolder && (
          <div className="mb-2 mt-2 ml-2 border-t-2 border-purple-200 dark:border-purple-800 relative">
            <div className="absolute -top-2.5 left-2 bg-white dark:bg-gray-950 px-2 text-[10px] text-purple-600 dark:text-purple-400 font-medium">
              代码生成文件
            </div>
          </div>
        )}
        <div
          className={`flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-900 rounded px-2 py-1 cursor-default ${isGenerated ? "bg-purple-50 dark:bg-purple-950/20" : ""
            }`}
          style={pad}
          onClick={(e) => {
            if (isDir) return toggleExpand(node.path);
            if (singleClickTimerRef.current) {
              window.clearTimeout(singleClickTimerRef.current);
              singleClickTimerRef.current = null;
            }
            singleClickTimerRef.current = window.setTimeout(() => {
              openNode(node);
              singleClickTimerRef.current = null;
            }, 180);
          }}
          onDoubleClick={(e) => {
            if (isDir) return;
            e.stopPropagation();
            if (singleClickTimerRef.current) {
              window.clearTimeout(singleClickTimerRef.current);
              singleClickTimerRef.current = null;
            }
            if (node.download_url) {
              downloadFileByUrl(node.name, node.download_url);
            } else {
              openNode(node);
            }
          }}
          onContextMenu={(e) => onContextMenu(e, node)}
          onDragOver={(e) => {
            if (isDir) e.preventDefault();
          }}
          onDrop={async (e) => {
            if (!isDir) return;
            e.preventDefault();
            const dt = e.dataTransfer;
            // 1) 如果是从 OS 拖入文件
            if (dt.files && dt.files.length) {
              uploadToDir(node.path, dt.files || []);
              return;
            }
            // 2) 如果是从 generated/ 内部拖动的文件，使用自定义 data 传递路径
            const srcPath = dt.getData("text/x-workspace-path");
            if (srcPath) {
              try {
                const url = `${API_CONFIG.BACKEND_BASE_URL
                  }/workspace/move?src=${encodeURIComponent(
                    srcPath
                  )}&dst_dir=${encodeURIComponent(
                    node.path
                  )}&session_id=${encodeURIComponent(sessionId)}`;
                const res = await fetch(url, { method: "POST" });
                if (res.ok) {
                  await loadWorkspaceTree();
                  await loadWorkspaceFiles();
                }
              } catch (err) {
                console.error("move error", err);
              }
            }
          }}
        >
          <div className="flex items-center gap-2 text-sm">
            {isDir ? (
              <>
                <span
                  className={
                    isGenerated
                      ? "text-purple-600 dark:text-purple-400"
                      : "text-gray-500"
                  }
                >
                  {expanded[node.path] ? "▾" : "▸"}
                </span>
                {isGenerated ? (
                  <Code2
                    className={`h-3.5 w-3.5 ${isGenerated
                      ? "text-purple-600 dark:text-purple-400"
                      : "text-gray-500"
                      }`}
                  />
                ) : (
                  <FolderOpen className="h-3.5 w-3.5 text-gray-500" />
                )}
              </>
            ) : (
              <span
                className={isGenerated ? "text-purple-400" : "text-gray-400"}
              >
                •
              </span>
            )}
            <span
              className={`truncate ${isGenerated
                ? "text-purple-700 dark:text-purple-300 font-medium"
                : ""
                }`}
            >
              {node.icon && !isGenerated ? `${node.icon} ` : ""}
              {node.name || "workspace"}
            </span>
            {!isDir && typeof node.size === "number" && (
              <span className="text-[10px] text-gray-400 ml-2 shrink-0">
                {formatFileSize(node.size)}
              </span>
            )}
            {isGenerated && !isDir && (
              <Sparkles className="h-3 w-3 text-purple-500 ml-1 shrink-0" />
            )}
          </div>
          {/* 双击/点击行为已经在容器上：目录展开，文件预览/下载保持一致 */}
        </div>
        {isDir && expanded[node.path] && node.children && (
          <div>{node.children.map((c) => renderTree(c, depth + 1))}</div>
        )}
      </div>
    );
  };

  const clearWorkspace = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(
        `${API_URLS.WORKSPACE_CLEAR}?session_id=${sessionId}`,
        {
          method: "DELETE",
        }
      );
      if (response.ok) {
        setWorkspaceFiles([]);
        await loadWorkspaceTree();
        await loadWorkspaceFiles();
        toast({
          description: "工作区已清空",
        });
      }
    } catch (error) {
      console.error("Failed to clear workspace:", error);
      toast({
        description: "清空失败",
        variant: "destructive",
      });
    }
  };

  const copyToClipboard = async (text: string): Promise<boolean> => {
    try {
      // 优先使用安全的 Clipboard API
      if (
        typeof navigator !== "undefined" &&
        (navigator as any).clipboard &&
        typeof (navigator as any).clipboard.writeText === "function"
      ) {
        await (navigator as any).clipboard.writeText(text);
        return true;
      }
    } catch (e) {
      // 继续尝试后备方案
    }
    try {
      // 后备方案：隐形 textarea + execCommand
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      textarea.style.pointerEvents = "none";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(textarea);
      return ok;
    } catch (e) {
      return false;
    }
  };

  const extractCode = (content: string): string => {
    const codeBlockMatch = content.match(/```(?:python)?\n?([\s\S]*?)```/);
    return codeBlockMatch ? codeBlockMatch[1].trim() : content;
  };

  const guessLanguageByExtension = (ext: string): string => {
    const e = ext.toLowerCase();
    const map: Record<string, string> = {
      js: "javascript",
      jsx: "jsx",
      ts: "typescript",
      tsx: "tsx",
      json: "json",
      py: "python",
      md: "markdown",
      html: "html",
      css: "css",
      sh: "bash",
      yml: "yaml",
      yaml: "yaml",
      csv: "csv",
      txt: "text",
      go: "go",
      rs: "rust",
      java: "java",
      php: "php",
      sql: "sql",
    };
    return map[e] || "text";
  };

  const normalizeToLocalFileUrl = (rawUrl: string): string => {
    const base =
      (API_CONFIG as any).FILE_SERVER_BASE || "http://localhost:8100";
    const safeBase = base.replace(/\/$/, "");

    if (!rawUrl) return safeBase;
    const trimmed = String(rawUrl).trim();

    // 绝对 http/https 链接：若是 localhost/127.* 或端口为 8100，则重写到 FILE_SERVER_BASE
    if (/^https?:\/\//i.test(trimmed)) {
      try {
        const u = new URL(trimmed);
        const needRewrite =
          u.hostname === "localhost" ||
          u.hostname.startsWith("127.") ||
          u.port === "8100";
        if (needRewrite) {
          const b = new URL(safeBase + "/");
          return `${b.origin}${b.pathname.replace(/\/$/, "")}${u.pathname}${u.search
            }${u.hash}`;
        }
        return trimmed;
      } catch {
        // fallthrough to relative handling
      }
    }

    // 处理以 // 开头的协议相对链接
    if (/^\/\//.test(trimmed)) {
      const proto =
        typeof window !== "undefined" ? window.location.protocol : "http:";
      return proto + trimmed;
    }

    // 去掉开头的 ./
    const rel = trimmed.replace(/^\.\//, "");

    // 如果以 /workspace/ 开头，接到文件服务器
    if (/^\/workspace\//.test(rel)) return `${safeBase}${rel}`;
    if (/^workspace\//.test(rel)) return `${safeBase}/${rel}`;

    // 其它相对路径或文件名，也认为位于文件服务器根目录
    return `${safeBase}/${rel.replace(/^\//, "")}`;
  };

  // 若 URL 缺少 generated 目录，则在 session 段后注入 /generated
  const ensureGeneratedInUrl = (url: string): string => {
    try {
      const u = new URL(url);
      // 仅处理指向文件服务器(8100)的链接
      if (!(u.hostname === "localhost" || u.hostname.startsWith("127."))) {
        return url;
      }
      // 路径形如 /session_xxx/xxx.png，则插入 /generated
      const parts = u.pathname.split("/").filter(Boolean);
      if (parts.length >= 2) {
        const [maybeSession, second] = parts;
        if (maybeSession.startsWith("session_") && second !== "generated") {
          const rest = parts.slice(1).join("/");
          u.pathname = `/${maybeSession}/generated/${rest}`;
          return u.toString();
        }
      }
      return url;
    } catch {
      return url;
    }
  };

  const openPreview = async (file: WorkspaceFile) => {
    setPreviewTitle(file.name);
    setPreviewDownloadUrl(file.download_url);
    setIsPreviewOpen(true);
    setPreviewLoading(true);

    const ext = (file.extension || "").toLowerCase();
    if (["png", "jpg", "jpeg", "gif", "svg", "webp"].includes(ext)) {
      setPreviewType("image");
      // 修正 URL
      const correctedUrl = ensureGeneratedInUrl(
        file.preview_url || file.download_url
      );
      setPreviewContent(correctedUrl);
      setPreviewLoading(false);
      return;
    }
    if (ext === "pdf") {
      setPreviewType("pdf");
      // 修正 URL
      const correctedUrl = ensureGeneratedInUrl(
        file.preview_url || file.download_url
      );
      setPreviewContent(correctedUrl);
      setPreviewLoading(false);
      return;
    }

    try {
      const normalized = normalizeToLocalFileUrl(
        file.preview_url || file.download_url
      );
      const target = ensureGeneratedInUrl(normalized);
      // 通过后端代理以避免 CORS
      const res = await fetch(
        `${API_CONFIG.BACKEND_BASE_URL}/proxy?url=${encodeURIComponent(target)}`
      );
      const contentType = res.headers.get("content-type") || "";
      if (!res.ok) throw new Error("failed to fetch preview");
      if (
        contentType.startsWith("text/") ||
        contentType.includes("json") ||
        contentType.includes("xml")
      ) {
        const text = await res.text();
        setPreviewType("text");
        setPreviewContent(text);
      } else {
        // 非文本直接提示下载/打开
        setPreviewType("binary");
        setPreviewContent(file.download_url);
      }
    } catch (e) {
      setPreviewType("binary");
      setPreviewContent(file.download_url);
    } finally {
      setPreviewLoading(false);
    }
  };

  useEffect(() => {
    if (isPreviewOpen && !previewLoading && previewScrollRef.current) {
      previewScrollRef.current.scrollTop = 0;
    }
  }, [isPreviewOpen, previewLoading, previewType, previewContent]);

  const handleDownload = async () => {
    try {
      if (previewType === "text" && typeof previewContent === "string") {
        const blob = new Blob([previewContent], {
          type: "text/plain;charset=utf-8",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = previewTitle || "file.txt";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        return;
      }

      const normalized = normalizeToLocalFileUrl(
        previewDownloadUrl || previewContent
      );
      const target = ensureGeneratedInUrl(normalized);
      const res = await fetch(
        `${API_CONFIG.BACKEND_BASE_URL}/proxy?url=${encodeURIComponent(target)}`
      );
      if (!res.ok) throw new Error("download failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = previewTitle || "download";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      const url = ensureGeneratedInUrl(previewDownloadUrl || previewContent);
      window.open(url, "_blank");
    }
  };

  const downloadFileByUrl = async (fileName: string, rawUrl: string) => {
    try {
      const normalized = normalizeToLocalFileUrl(rawUrl);
      const target = ensureGeneratedInUrl(normalized);
      const res = await fetch(
        `${API_CONFIG.BACKEND_BASE_URL}/proxy?url=${encodeURIComponent(target)}`
      );
      if (!res.ok) throw new Error("download failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName || "download";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      const fallbackUrl = ensureGeneratedInUrl(rawUrl);
      window.open(fallbackUrl, "_blank");
    }
  };

  const executeCode = async () => {
    setIsExecutingCode(true);
    try {
      const response = await fetch(API_URLS.EXECUTE_CODE, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code: codeEditorContent,
          session_id: sessionId,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setCodeExecutionResult(data.result);
        await loadWorkspaceFiles(); // Refresh file list after execution
      } else {
        setCodeExecutionResult("Error: Failed to execute code");
      }
    } catch (error) {
      setCodeExecutionResult(`Error: ${error}`);
    } finally {
      setIsExecutingCode(false);
    }
  };

  const renderMarkdownContent = useCallback((
    content: string,
    options?: { withinSection?: boolean }
  ) => {
    const withinSection = options?.withinSection ?? false;
    // 先处理代码块，将其分离出来
    const parts = content.split(/(```[\w]*\n[\s\S]*?```)/g);

    return (
      <div className="prose prose-sm max-w-none dark:prose-invert break-words [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5">
        {parts.map((part, index) => {
          // 检查是否是代码块
          const codeBlockMatch = part.match(/```(\w+)?\n([\s\S]*?)```/);
          if (codeBlockMatch) {
            const [, language, code] = codeBlockMatch;
            return (
              <CodeBlockView
                key={index}
                language={language || "python"}
                code={code}
                showHeader={!withinSection}
                isDarkMode={isDarkMode}
                onEdit={(c) => {
                  setCodeEditorContent(c);
                  setSelectedCodeSection(c);
                  setShowCodeEditor(true);
                }}
              />
            );
          }

          // 处理普通 markdown 内容
          if (part.trim()) {
            return (
              <ReactMarkdown
                key={index}
                remarkPlugins={[remarkGfm]}
                components={{
                  code: ({ children, ...props }: any) => (
                    <code
                      className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm font-mono"
                      {...props}
                    >
                      {children}
                    </code>
                  ),
                  h1: ({ children }) => (
                    <h1 className="text-2xl font-bold mt-4 mb-2">{children}</h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-xl font-semibold mt-4 mb-2">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-lg font-semibold mt-4 mb-2">
                      {children}
                    </h3>
                  ),
                  a: ({ href, children }) => {
                    const normalized = normalizeToLocalFileUrl(
                      String(href || "")
                    );
                    const corrected = ensureGeneratedInUrl(normalized);
                    const proxied = `${API_CONFIG.BACKEND_BASE_URL
                      }/proxy?url=${encodeURIComponent(corrected)}`;
                    return (
                      <a
                        href={proxied}
                        className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {children}
                      </a>
                    );
                  },
                  img: ({ src, alt }: any) => {
                    const normalizedSrc = normalizeToLocalFileUrl(src || "");
                    const correctedSrc = ensureGeneratedInUrl(normalizedSrc);
                    const proxiedSrc = `${API_CONFIG.BACKEND_BASE_URL
                      }/proxy?url=${encodeURIComponent(correctedSrc)}`;
                    return (
                      <img
                        src={proxiedSrc}
                        alt={alt || ""}
                        className="max-w-full h-auto rounded-lg my-2"
                      />
                    );
                  },
                  ol: ({ children }) => (
                    <ol className="list-decimal pl-5 space-y-1">{children}</ol>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc pl-5 space-y-1">{children}</ul>
                  ),
                }}
              >
                {part}
              </ReactMarkdown>
            );
          }

          return null;
        })}
      </div>
    );
  }, [isDarkMode]);

  const renderSectionContent = useCallback(
    (content: string) => {
      return renderMarkdownContent(content, { withinSection: true });
    },
    [renderMarkdownContent]
  );

  // 解析 Markdown 中的文件/图片链接，返回用于卡片渲染的数据
  const parseGeneratedFiles = (
    content: string
  ): Array<{ name: string; url: string; isImage: boolean }> => {
    const result: { name: string; url: string; isImage: boolean }[] = [];
    let m: RegExpExecArray | null;
    // 1) 列表形如: - [name](url)
    const linkRe = /\- \[(.*?)\]\((.*?)\)/g;
    while ((m = linkRe.exec(content)) !== null) {
      const name = m[1];
      const url = normalizeToLocalFileUrl(m[2]);
      const isImage = /\.(png|jpg|jpeg|gif|webp|svg)(\?.*)?$/i.test(url);
      result.push({ name, url, isImage });
    }
    // 2) 图片 Markdown: ![name](url)
    const imgRe = /!\[(.*?)\]\((.*?)\)/g;
    while ((m = imgRe.exec(content)) !== null) {
      const name = m[1];
      const url = normalizeToLocalFileUrl(m[2]);
      result.push({ name, url, isImage: true });
    }
    // 3) 兜底：文中出现的裸链接
    const urlRe = /(https?:\/\/[^\s)]+)/g;
    while ((m = urlRe.exec(content)) !== null) {
      const url = normalizeToLocalFileUrl(m[1]);
      const isImage = /\.(png|jpg|jpeg|gif|webp|svg)(\?.*)?$/i.test(url);
      if (isImage)
        result.push({ name: url.split("/")?.pop() || "image", url, isImage });
    }
    // 去重同 url
    const seen = new Set<string>();
    return result.filter((f) =>
      seen.has(f.url) ? false : (seen.add(f.url), true)
    );
  };

  // 提取消息中的所有步骤
  const extractSections = (content: string, messageIndex?: number) => {
    const sectionConfigs = {
      Analyze: { icon: "🔍", color: "bg-blue-500" },
      Understand: { icon: "🧠", color: "bg-cyan-500" },
      Code: { icon: "💻", color: "bg-gray-500" },
      Execute: { icon: "⚡", color: "bg-orange-500" },
      Answer: { icon: "✅", color: "bg-green-500" },
      File: { icon: "📎", color: "bg-purple-500" }, // 添加 File 类型
    };

    const allMatches: Array<{
      type: keyof typeof sectionConfigs;
      position: number;
    }> = [];

    Object.keys(sectionConfigs).forEach((type) => {
      const regex = new RegExp(`<${type}>([\\s\\S]*?)</${type}>`, "g");
      let match;

      while ((match = regex.exec(content)) !== null) {
        allMatches.push({
          type: type as keyof typeof sectionConfigs,
          position: match.index,
        });
      }
    });

    // 按位置排序，然后生成 sectionKey（与 renderMessageWithSections 逻辑一致）
    allMatches.sort((a, b) => a.position - b.position);

    return allMatches.map((m, index) => ({
      type: m.type,
      sectionKey:
        messageIndex !== undefined
          ? `msg${messageIndex}-${m.type}-${index}` // 包含消息索引
          : `${m.type}-${index}`, // 兼容旧逻辑
      config: sectionConfigs[m.type],
    }));
  };

  // 滚动到指定步骤
  const scrollToSection = (sectionKey: string) => {
    const container = messagesContainerRef.current;
    if (!container) {
      console.warn("Container not found");
      return;
    }

    // 展开目标块（如果它是折叠的）
    setCollapsedSections((prev) => {
      const next = { ...prev };
      // 提取 baseKey（去掉 msg{index}- 前缀）
      const baseKey = sectionKey.replace(/^msg\d+-/, "");

      // 如果该块是折叠的，则展开它（同时更新两种格式的 key）
      if (prev[sectionKey] || prev[baseKey]) {
        next[sectionKey] = false;
        next[baseKey] = false;
        return next;
      }
      return prev;
    });

    // 标记为手动操作，防止自动折叠覆盖
    setManualLocks((prev) => {
      const baseKey = sectionKey.replace(/^msg\d+-/, "");
      return {
        ...prev,
        [sectionKey]: true,
        [baseKey]: true,
      };
    });

    // 使用延迟确保 DOM 已更新和展开动画完成
    setTimeout(() => {
      const element = document.querySelector(
        `[data-section-key="${sectionKey}"]`
      );

      if (!element) {
        console.warn(`Element with key ${sectionKey} not found`);
        return;
      }

      const elementRect = element.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      const scrollTop = container.scrollTop;

      // 计算目标滚动位置（居中显示）
      const targetScroll =
        scrollTop +
        elementRect.top -
        containerRect.top -
        containerRect.height / 2 +
        elementRect.height / 2;

      container.scrollTo({
        top: Math.max(0, targetScroll),
        behavior: "smooth",
      });

      setActiveSection(sectionKey);
    }, 150);
  };

  const updateActiveSectionFromScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const sections = document.querySelectorAll("[data-section-key]");
    const containerRect = container.getBoundingClientRect();
    const containerMiddle = containerRect.top + containerRect.height / 2;

    let closestSection = "";
    let closestDistance = Infinity;

    sections.forEach((section) => {
      const rect = section.getBoundingClientRect();
      const sectionMiddle = rect.top + rect.height / 2;
      const distance = Math.abs(sectionMiddle - containerMiddle);

      // 找到离容器中心最近的 section
      if (
        distance < closestDistance &&
        rect.top < containerRect.bottom &&
        rect.bottom > containerRect.top
      ) {
        closestDistance = distance;
        closestSection = section.getAttribute("data-section-key") || "";
      }
    });

    if (closestSection) {
      setActiveSection(closestSection);
    }
  }, []);

  // 监听滚动，更新当前激活的步骤（避免 messages 更新时反复解绑/绑定 scroll 事件）
  const activeSectionRafRef = useRef<number | null>(null);
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const onScroll = () => {
      // 只有用户当前在底部时才自动跟随输出
      const distanceToBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight;
      stickToBottomRef.current = distanceToBottom <= 24;

      if (activeSectionRafRef.current) return;
      activeSectionRafRef.current = window.requestAnimationFrame(() => {
        activeSectionRafRef.current = null;
        updateActiveSectionFromScroll();
      });
    };

    onScroll(); // 初始化
    container.addEventListener("scroll", onScroll);
    return () => {
      container.removeEventListener("scroll", onScroll);
      if (activeSectionRafRef.current) {
        window.cancelAnimationFrame(activeSectionRafRef.current);
        activeSectionRafRef.current = null;
      }
    };
  }, [updateActiveSectionFromScroll]);

  // 新消息追加/清空时刷新一次 active section（不在流式内容每次变化时都跑）
  useEffect(() => {
    if (!messagesContainerRef.current) return;
    window.requestAnimationFrame(() => updateActiveSectionFromScroll());
  }, [messages.length, updateActiveSectionFromScroll]);

  // 流式阶段的轻量渲染：支持 <Analyze>/<Code> 等块，但避免高开销的 Markdown/高亮解析
  const renderMessageWithSectionsStreaming = useCallback(
    (content: string, messageIndex?: number) => {
      const sectionTypes = [
        "Analyze",
        "Understand",
        "Code",
        "Execute",
        "Answer",
        "File",
      ] as const;
      const sectionConfigs: Record<
        (typeof sectionTypes)[number],
        { icon: string; color: string }
      > = {
        Analyze: {
          icon: "🔍",
          color:
            "bg-blue-50 border-blue-200 dark:bg-blue-950/30 dark:border-blue-800",
        },
        Understand: {
          icon: "🧠",
          color:
            "bg-cyan-50 border-cyan-200 dark:bg-cyan-950/30 dark:border-cyan-800",
        },
        Code: {
          icon: "💻",
          color:
            "bg-gray-50 border-gray-200 dark:bg-gray-950/30 dark:border-gray-700",
        },
        Execute: {
          icon: "⚡",
          color:
            "bg-orange-50 border-orange-200 dark:bg-orange-950/30 dark:border-orange-800",
        },
        Answer: {
          icon: "✅",
          color:
            "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-800",
        },
        File: {
          icon: "📎",
          color:
            "bg-purple-50 border-purple-200 dark:bg-purple-950/30 dark:border-purple-800",
        },
      };

      // 没有结构化标签时，保持最轻量文本渲染（避免每个 chunk 都触发 Markdown/高亮重解析）
      if (!content.includes("<")) {
        return (
          <div className="text-sm break-words whitespace-pre-wrap">
            {content}
          </div>
        );
      }

      const parts: React.ReactNode[] = [];
      const openRe = /<(Analyze|Understand|Code|Execute|Answer|File)>/g;
      let cursor = 0;
      let sectionIndex = 0;
      let m: RegExpExecArray | null;

      while ((m = openRe.exec(content)) !== null) {
        const type = m[1] as StructuredSectionType;
        const start = m.index;

        if (start > cursor) {
          const before = content.slice(cursor, start);
          parts.push(
            <StreamingMarkdownBlock
              key={`stream-md-${cursor}`}
              className="markdown-content mb-2"
              content={before}
              renderMarkdownContent={renderMarkdownContent}
            />
          );
        }

        const openTag = m[0];
        const openEnd = start + openTag.length;
        const closeTag = `</${type}>`;
        const closeIdx = content.indexOf(closeTag, openEnd);
        const isComplete = closeIdx !== -1;
        const bodyEnd = isComplete ? closeIdx : content.length;
        const body = content.slice(openEnd, bodyEnd).trim();

        const baseKey = `${type}-${sectionIndex}`;
        const msgKey =
          messageIndex !== undefined ? `msg${messageIndex}-${type}-${sectionIndex}` : baseKey;
        const sectionKey = msgKey;
        const isCollapsed =
          (collapsedSections as any)[msgKey] ??
          (collapsedSections as any)[baseKey] ??
          false;

        const toggleSection = () => {
          setCollapsedSections((prev) => {
            const next = { ...prev } as Record<string, boolean>;
            const current = (prev as any)[msgKey] ?? (prev as any)[baseKey] ?? false;
            next[msgKey] = !current;
            next[baseKey] = !current;
            return next;
          });
        };

        parts.push(
          <div
            key={`stream-section-${sectionKey}`}
            className={`mb-4 border rounded-lg overflow-hidden ${sectionConfigs[type].color}`}
            data-section={type}
            data-section-key={sectionKey}
          >
            <div className="flex items-center justify-between px-3 py-2 bg-white/60 dark:bg-black/30 border-b border-black/5 dark:border-white/10">
              <div className="flex items-center gap-2 min-w-0">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleSection}
                  className="h-5 w-5 p-0 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  {isCollapsed ? (
                    <ChevronRight className="h-3 w-3" />
                  ) : (
                    <ChevronDown className="h-3 w-3" />
                  )}
                </Button>
                <span className="text-sm">{sectionConfigs[type].icon}</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {type}
                </span>
                {!isComplete && (
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    （生成中）
                  </span>
                )}
              </div>
            </div>
            {!isCollapsed && (
              <div className="p-3">
                <StreamingSectionBody
                  type={type}
                  content={body}
                  isComplete={isComplete}
                  renderSectionContent={renderSectionContent}
                />
              </div>
            )}
          </div>
        );

        sectionIndex += 1;
        cursor = isComplete ? closeIdx + closeTag.length : content.length;
        openRe.lastIndex = cursor;

        if (!isComplete) break;
      }

      if (cursor < content.length) {
        const after = content.slice(cursor);
        if (after.trim()) {
          parts.push(
            <div key="stream-text-end" className="text-sm break-words whitespace-pre-wrap">
              {after}
            </div>
          );
        }
      }

      if (parts.length === 0) {
        return (
          <div className="text-sm break-words whitespace-pre-wrap">
            {content}
          </div>
        );
      }

      return <>{parts}</>;
    },
    [collapsedSections, renderMarkdownContent, renderSectionContent]
  );

  const renderMessageWithSections = useCallback((
    content: string,
    messageIndex?: number
  ) => {
    const sectionConfigs = {
      Analyze: {
        icon: "🔍",
        color:
          "bg-blue-50 border-blue-200 dark:bg-blue-950/30 dark:border-blue-800",
      },
      Understand: {
        icon: "🧠",
        color:
          "bg-cyan-50 border-cyan-200 dark:bg-cyan-950/30 dark:border-cyan-800",
      },
      Code: {
        icon: "💻",
        color:
          "bg-gray-50 border-gray-200 dark:bg-gray-950/30 dark:border-gray-700",
      },
      Execute: {
        icon: "⚡",
        color:
          "bg-orange-50 border-orange-200 dark:bg-orange-950/30 dark:border-orange-800",
      },
      Answer: {
        icon: "✅",
        color:
          "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-800",
      },
      File: {
        icon: "📎",
        color:
          "bg-purple-50 border-purple-200 dark:bg-purple-950/30 dark:border-purple-800",
      },
    };

    // 首先分割内容，找出所有标签
    const allMatches: Array<{
      type: keyof typeof sectionConfigs;
      content: string;
      position: number;
      fullMatch: string;
    }> = [];

    Object.keys(sectionConfigs).forEach((type) => {
      // 使用 [\s\S]*? 以兼容不支持 s 标志的环境
      const regex = new RegExp(`<${type}>([\\s\\S]*?)</${type}>`, "g");
      let match;

      while ((match = regex.exec(content)) !== null) {
        allMatches.push({
          type: type as keyof typeof sectionConfigs,
          content: match[1].trim(),
          position: match.index,
          fullMatch: match[0],
        });
      }
    });

    // 如果没有找到结构化标签，渲染为 Markdown
    if (allMatches.length === 0) {
      return (
        <div className="markdown-content">{renderMarkdownContent(content)}</div>
      );
    }

    // 按位置排序
    allMatches.sort((a, b) => a.position - b.position);

    const parts = [];
    let lastPosition = 0;

    allMatches.forEach((match, index) => {
      // 添加标签前的普通文本
      if (match.position > lastPosition) {
        const beforeText = content.slice(lastPosition, match.position);
        if (beforeText.trim()) {
          parts.push(
            <div key={`text-${index}`} className="markdown-content mb-2">
              {renderMarkdownContent(beforeText)}
            </div>
          );
        }
      }

      // 添加结构化标签
      const config = sectionConfigs[match.type];
      const baseKey = `${match.type}-${index}`;
      const msgKey =
        messageIndex !== undefined
          ? `msg${messageIndex}-${match.type}-${index}`
          : baseKey;
      const sectionKey = msgKey;
      const isCollapsed =
        (collapsedSections as any)[msgKey] ??
        (collapsedSections as any)[baseKey] ??
        false;

      const toggleSection = () => {
        setCollapsedSections((prev) => {
          const next = { ...prev } as Record<string, boolean>;
          const current =
            (prev as any)[msgKey] ?? (prev as any)[baseKey] ?? false;
          next[msgKey] = !current;
          next[baseKey] = !current;
          return next;
        });
        setManualLocks((prev) => ({
          ...prev,
          [msgKey]: true,
          [baseKey]: true,
        }));
      };

      // 如果是 File 标签，解析其中的链接为卡片
      let sectionBody = match.content;
      let fileGallery: JSX.Element | null = null;
      if (match.type === "File") {
        const files = parseGeneratedFiles(match.content);
        if (files.length) {
          fileGallery = (
            <div className="mt-3">
              <div className="text-xs text-gray-500 mb-2">相关文件</div>
              <div className="grid grid-cols-2 gap-2">
                {files.map((f, i) => {
                  // 通过代理访问图片，并自动修正缺少 generated 的 URL
                  const correctedUrl = ensureGeneratedInUrl(f.url);
                  const proxiedUrl = `${API_CONFIG.BACKEND_BASE_URL
                    }/proxy?url=${encodeURIComponent(correctedUrl)}`;
                  return (
                    <div
                      key={i}
                      className="border border-gray-200 dark:border-gray-700 rounded overflow-hidden bg-white dark:bg-black"
                    >
                      {f.isImage ? (
                        <a href={proxiedUrl} target="_blank" rel="noreferrer">
                          <img
                            src={proxiedUrl}
                            alt={f.name}
                            className="w-full h-28 object-contain bg-white dark:bg-black"
                          />
                        </a>
                      ) : (
                        <a
                          href={proxiedUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="block p-2 text-xs truncate hover:bg-gray-50 dark:hover:bg-gray-900"
                        >
                          {f.name}
                        </a>
                      )}
                      <div className="flex items-center justify-between px-2 py-1 border-t border-gray-200 dark:border-gray-800">
                        <div className="text-[10px] truncate max-w-[70%] text-gray-500">
                          {f.name}
                        </div>
                        <a
                          href={proxiedUrl}
                          download
                          className="text-[10px] text-blue-600 hover:underline"
                        >
                          下载
                        </a>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        }
      }

      parts.push(
        <div
          key={`section-${index}`}
          className="mb-4 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
          data-section={match.type}
          data-section-key={sectionKey}
        >
          <div className="flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleSection}
                className="h-5 w-5 p-0 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                {isCollapsed ? (
                  <ChevronRight className="h-3 w-3" />
                ) : (
                  <ChevronDown className="h-3 w-3" />
                )}
              </Button>
              <span className="text-sm">{config.icon}</span>
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {match.type}
              </span>
            </div>
            <div className="flex items-center gap-1">
              {match.type === "Answer" && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={async () => {
                    if (isTyping) {
                      toast({
                        description: "执行中，暂时无法导出",
                        variant: "destructive",
                      });
                      return;
                    }
                    await exportReportBackendRef.current();
                  }}
                  className="h-5 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  title="后端导出 PDF/MD 到 workspace"
                >
                  <Download className="h-3 w-3" />
                </Button>
              )}
              {(match.type === "Code" ||
                match.type === "Analyze" ||
                match.type === "Understand") && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={async () => {
                        const text =
                          match.type === "Code"
                            ? extractCode(match.content)
                            : match.content;
                        const ok = await copyToClipboard(text.trim());
                        toast({
                          description: ok ? "已复制" : "复制失败",
                          variant: ok ? undefined : "destructive",
                        });
                      }}
                      className="h-5 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                    {match.type === "Code" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const code = extractCode(match.content);
                          setCodeEditorContent(code);
                          setSelectedCodeSection(match.content);
                          setShowCodeEditor(true);
                        }}
                        className="h-5 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                      >
                        <Edit className="h-3 w-3" />
                      </Button>
                    )}
                  </>
                )}
              {match.type === "Execute" && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      const executionOutput = extractCode(
                        sectionBody || match.content || ""
                      );
                      const textToCopy = executionOutput || sectionBody || "";
                      if (textToCopy.trim()) {
                        const ok = await copyToClipboard(textToCopy.trim());
                        toast({
                          description: ok ? "已复制" : "复制失败",
                          variant: ok ? undefined : "destructive",
                        });
                      }
                    }}
                    className="h-5 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    title="复制此 Execute 的输出"
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                </>
              )}
            </div>
          </div>
          {!isCollapsed && (
            <div
              className={`p-3 ${match.type === "Answer" ? "answer-body" : ""}`}
            >
              {renderSectionContent(sectionBody)}
              {fileGallery}
            </div>
          )}
        </div>
      );

      lastPosition = match.position + match.fullMatch.length;
    });

    // 添加最后剩余的文本
    if (lastPosition < content.length) {
      const afterText = content.slice(lastPosition);
      if (afterText.trim()) {
        parts.push(
          <div key="text-end" className="markdown-content mt-2">
            {renderMarkdownContent(afterText)}
          </div>
        );
      }
    }

    return <>{parts}</>;
  }, [collapsedSections, isTyping, renderMarkdownContent, renderSectionContent, toast]);

  // 根据完整内容自动折叠：除最后一个块外全部折叠
  const autoCollapseForContent = useCallback(
    (content: string, messageIndex?: number) => {
      if (!autoCollapseEnabled) return;
      const sectionTypes = [
        "Analyze",
        "Understand",
        "Code",
        "Execute",
        "File",
        "Answer",
      ] as const;
      const matches: Array<{ type: string; index: number; pos: number }> = [];
      sectionTypes.forEach((t) => {
        const re = new RegExp(`<${t}>([\\s\\S]*?)</${t}>`, "g");
        let m: RegExpExecArray | null;
        let local = 0;
        while ((m = re.exec(content)) !== null) {
          matches.push({ type: t, index: local++, pos: m.index });
        }
      });
      if (matches.length === 0) return;
      matches.sort((a, b) => a.pos - b.pos);
      const next: Record<string, boolean> = {};
      matches.forEach((m, i) => {
        const baseKey = `${m.type}-${i}`;
        const msgKey =
          messageIndex !== undefined ? `msg${messageIndex}-${m.type}-${i}` : null;
        const key = msgKey || baseKey;
        next[key] = i !== matches.length - 1; // 最后一个不折叠
      });
      setCollapsedSections((prev) => {
        const merged: Record<string, boolean> = { ...prev };
        // 只在未手动锁定的 key 上更新，保留用户手动状态
        for (const key in next) {
          const baseKey = key.replace(/^msg\d+-/, "");
          if (!manualLocks[key] && !manualLocks[baseKey]) merged[key] = next[key];
        }
        return merged;
      });
    },
    [autoCollapseEnabled, manualLocks]
  );

  const handleSendMessage = async () => {
    if (!inputValue.trim() && attachments.length === 0) return;
    const baseMessageIndex = messages.length;
    const aiMessageIndex = baseMessageIndex + 1;

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: "user",
      timestamp: new Date(),
      attachments: attachments.length > 0 ? [...attachments] : undefined,
    };

    setMessages((prev) => [...prev, newMessage]);
    setInputValue("");
    setAttachments([]);
    setIsTyping(true);

    try {
      const response = await fetch(API_URLS.CHAT_COMPLETIONS, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "DeepAnalyze-8B", // 修正模型名
          messages: [
            ...messages
              .filter((m) => !m.localOnly)
              .map((msg) => ({
                role: msg.sender === "user" ? "user" : "assistant",
                content: msg.content,
              })),
            {
              role: "user",
              content: inputValue,
            },
          ],
          stream: true, // [修改] 明确开启流式模式
          session_id: sessionId,
        }),
      });

      const contentType = response.headers.get("content-type") || "";
      console.log("[Chat] status=", response.status, "ctype=", contentType);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // 情况1: 非流式 JSON (兜底)
      if (contentType.includes("application/json")) {
        const data = await response.json();
        const content = data?.choices?.[0]?.message?.content || "";
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            sender: "ai",
            content,
            timestamp: new Date(),
          },
        ]);
        autoCollapseForContent(content, aiMessageIndex);
        if (content.includes("<File>")) {
          await loadWorkspaceTree();
          await loadWorkspaceFiles();
        }
        setIsTyping(false);
        return;
      }

      // 情况2: 流式响应 (NDJSON / SSE)
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) {
        setIsTyping(false);
        setStreamingMessageId(null);
        return;
      }

      // 预先插入 AI 消息占位
      const aiMsgId = `${Date.now()}-${Math.random()}`;
      setStreamingMessageId(aiMsgId);
      setMessages((prev) => [
        ...prev,
        {
          id: aiMsgId,
          sender: "ai",
          content: "",
          timestamp: new Date(),
        },
      ]);

      aiPendingContentRef.current = "";
      aiDisplayedContentRef.current = "";

      if (streamRafRef.current) {
        cancelAnimationFrame(streamRafRef.current);
        streamRafRef.current = null;
      }

      // [修改] 用于在本地累积完整的消息内容
      let accumulatedMessage = "";

      // 更新 UI 的辅助函数
      const flushAiMessage = (visibleText: string) => {
        setMessages((prev) => {
          const next = [...prev];
          const idx = next.findIndex((m) => m.id === aiMsgId);
          if (idx >= 0) {
            next[idx] = { ...next[idx], content: visibleText };
          }
          return next;
        });

        if (visibleText.includes("<File>")) {
          if (fileRefreshTimerRef.current) {
            window.clearTimeout(fileRefreshTimerRef.current);
          }
          fileRefreshTimerRef.current = window.setTimeout(async () => {
            await loadWorkspaceTree();
            await loadWorkspaceFiles();
            fileRefreshTimerRef.current = null;
          }, 300);
        }
      };

      // 启动平滑动画循环
      const loop = () => {
        const pending = aiPendingContentRef.current;
        const displayed = aiDisplayedContentRef.current;

        if (displayed !== pending) {
          const diff = pending.length - displayed.length;
          // 若 pending 比 displayed 短（理论不应发生），或差异极小，则直接同步
          if (diff < 0) {
            aiDisplayedContentRef.current = pending;
            flushAiMessage(pending);
          } else {
            // 自适应速度：
            // 如果落后很多（网络卡顿后突然涌入），则步进大一些以快速追赶
            // 如果落后很少，则步进小，实现打字机效果
            // min=1 保证不卡死，max 限制瞬时渲染量
            // Math.ceil(diff / 10) 意味着每帧追赶 10% 的差距 -> 渐进式平滑
            const step = Math.max(1, Math.ceil(diff / 5));

            const next = pending.slice(0, displayed.length + step);
            aiDisplayedContentRef.current = next;
            flushAiMessage(next);


          }
        }
        streamRafRef.current = requestAnimationFrame(loop);
      };
      streamRafRef.current = requestAnimationFrame(loop);

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed === "data: [DONE]") continue;

          try {
            const json = JSON.parse(trimmed);
            const deltaContent = json.choices?.[0]?.delta?.content;

            if (deltaContent) {
              accumulatedMessage += deltaContent;
              // 仅更新 pending，不直接刷新 UI
              aiPendingContentRef.current = accumulatedMessage;
            }
          } catch (e) {
            console.warn("JSON parse error for line:", trimmed, e);
          }
        }
      }

      if (buffer.trim()) {
        try {
          const json = JSON.parse(buffer.trim());
          const deltaContent = json.choices?.[0]?.delta?.content;
          if (deltaContent) {
            accumulatedMessage += deltaContent;
            aiPendingContentRef.current = accumulatedMessage;
          }
        } catch (e) { }
      }

      // 流束后，确保最终内容完全显示
      // 停止动画循环
      if (streamRafRef.current) {
        cancelAnimationFrame(streamRafRef.current);
        streamRafRef.current = null;
      }
      // 强制同步最后状态
      flushAiMessage(accumulatedMessage);
      autoCollapseForContent(accumulatedMessage, aiMessageIndex);

      // 结束后刷新一次文件列表确保无遗漏
      await loadWorkspaceFiles();
      await loadWorkspaceTree();
      setIsTyping(false); // 结束加载状态
      setStreamingMessageId(null);

    } catch (error) {
      console.error("Error sending message:", error);
      setIsTyping(false);
      setStreamingMessageId(null);
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = event.target.files;
    if (!files) return;
    await uploadToDir("", files);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((att) => att.id !== id));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <>
      <div
        className="h-screen bg-white dark:bg-black text-black dark:text-white"
        suppressHydrationWarning
      >
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* Left Panel - Workspace Tree */}
          <ResizablePanel defaultSize={25} minSize={15}>
            <div className="flex flex-col min-h-0 min-w-0 h-full">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800 h-12">
                <h2 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Files
                </h2>
                <div
                  className="flex items-center gap-1"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={async (e) => {
                    e.preventDefault();
                    const items = Array.from(e.dataTransfer.files || []);
                    if (!items.length) return;
                    const form = new FormData();
                    items.forEach((f) => form.append("files", f));
                    const dir = contextTarget?.is_dir ? contextTarget.path : "";
                    try {
                      const url = `${API_URLS.WORKSPACE_UPLOAD_TO
                        }?dir=${encodeURIComponent(
                          dir
                        )}&session_id=${encodeURIComponent(sessionId)}`;
                      await fetch(url, { method: "POST", body: form });
                      await loadWorkspaceTree();
                      await loadWorkspaceFiles();
                    } catch (err) {
                      console.error(err);
                    }
                  }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    onChange={handleFileUpload}
                    className="hidden"
                    accept="*"
                  />
                  {/* <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                    className="h-6 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-500 dark:hover:text-gray-300"
                  >
                    <Paperclip className="h-3 w-3" />
                  </Button> */}
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-500 dark:hover:text-gray-300"
                        title="清空 workspace"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>清空 workspace？</AlertDialogTitle>
                        <AlertDialogDescription>
                          将删除 workspace
                          根目录下的所有文件与文件夹，此操作不可撤销。
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>取消</AlertDialogCancel>
                        <AlertDialogAction
                          className="bg-red-600 hover:bg-red-700"
                          onClick={clearWorkspace}
                        >
                          确认清空
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </div>

              <div
                ref={treeContainerRef}
                className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden pl-3 pr-1 py-2"
              >
                <div
                  className={`mb-2 rounded border border-dashed flex items-center justify-center h-20 text-xs select-none ${dropActive
                    ? "bg-blue-50 border-blue-300 text-blue-600"
                    : "bg-gray-50 dark:bg-gray-900/40 border-gray-300 dark:border-gray-700 text-gray-500"
                    }`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDropActive(true);
                  }}
                  onDragLeave={() => setDropActive(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDropActive(false);
                    const files = e.dataTransfer.files;
                    if (files && files.length) uploadToDir("", files);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {/* 独立隐藏 input 兼容点击上传 */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    onChange={handleFileUpload}
                    className="hidden"
                    accept="*"
                  />
                  <div className="flex items-center gap-2">
                    <Upload className="h-4 w-4" />
                    <span>拖拽或点击此处上传（workspace 根目录）</span>
                  </div>
                </div>
                {uploadMsg && (
                  <div className="px-2 pb-2 text-[11px] text-gray-500">
                    {uploadMsg}
                  </div>
                )}
                {workspaceTree ? (
                  <Tree
                    width={treeSize.w || 300}
                    height={treeSize.h || 400}
                    data={toArbor(workspaceTree).children || []}
                    openByDefault
                    indent={14}
                    rowHeight={28}
                  >
                    {Row}
                  </Tree>
                ) : (
                  <div className="flex items-center justify-center h-full text-sm text-gray-500">
                    Loading...
                  </div>
                )}
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Middle Panel - Chat & Analysis */}
          <ResizablePanel defaultSize={40} minSize={25}>
            <div className="flex flex-col min-h-0 min-w-0 h-full">
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800 h-12 shrink-0">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <h1 className="text-sm font-medium">Assistant</h1>
                    {isTyping && (
                      <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                        <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                        <span>执行中…</span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                    <span>自动折叠</span>
                    <Switch
                      className="data-[state=unchecked]:bg-gray-200 data-[state=unchecked]:border data-[state=unchecked]:border-gray-300"
                      checked={autoCollapseEnabled}
                      onCheckedChange={(v: boolean) => {
                        setAutoCollapseEnabled(!!v);
                        if (typeof window !== "undefined") {
                          localStorage.setItem(
                            "autoCollapseEnabled",
                            (!!v).toString()
                          );
                        }
                        // 关闭自动折叠时，展开所有块
                        if (!v) {
                          setCollapsedSections({});
                          setManualLocks({});
                        }
                      }}
                    />
                  </div>
                  {/* 旧菜单已移除 */}
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={toggleTheme}
                    className="h-8 w-8 p-0"
                  >
                    {mounted ? (
                      isDarkMode ? (
                        <Sun className="h-4 w-4" />
                      ) : (
                        <Moon className="h-4 w-4" />
                      )
                    ) : (
                      <Moon className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Step Navigator - Top Horizontal */}
              {(() => {
                // 只显示最后一条 AI 消息的步骤
                let lastAiMsgIndex = -1;
                let lastAiMsg = null;

                for (let i = messages.length - 1; i >= 0; i--) {
                  if (messages[i].sender === "ai") {
                    lastAiMsg = messages[i];
                    lastAiMsgIndex = i;
                    break;
                  }
                }

                if (!lastAiMsg || lastAiMsgIndex === -1) return null;

                const allSections = extractSections(
                  lastAiMsg.content,
                  lastAiMsgIndex
                );

                if (allSections.length === 0) return null;

                return (
                  <div className="relative border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-950 px-6 py-4 overflow-hidden">
                    {/* 背景装饰 */}
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-50/50 via-purple-50/30 to-pink-50/50 dark:from-blue-950/20 dark:via-purple-950/10 dark:to-pink-950/20 pointer-events-none" />

                    <div
                      ref={stepNavigatorRef}
                      className="relative flex items-center gap-1 overflow-x-auto pb-1 scrollbar-thin"
                    >
                      {allSections.map((section, idx) => {
                        const isActive = activeSection === section.sectionKey;
                        const activeIdx = allSections.findIndex(
                          (s) => s.sectionKey === activeSection
                        );
                        const isCompleted = activeIdx > idx;
                        const isPending = activeIdx < idx;

                        // 颜色映射
                        const colorMap: Record<
                          string,
                          {
                            bg: string;
                            border: string;
                            glow: string;
                            text: string;
                          }
                        > = {
                          "bg-blue-500": {
                            bg: "bg-blue-500",
                            border: "border-blue-400",
                            glow: "shadow-blue-500/50",
                            text: "text-blue-600",
                          },
                          "bg-cyan-500": {
                            bg: "bg-cyan-500",
                            border: "border-cyan-400",
                            glow: "shadow-cyan-500/50",
                            text: "text-cyan-600",
                          },
                          "bg-gray-500": {
                            bg: "bg-gray-500",
                            border: "border-gray-400",
                            glow: "shadow-gray-500/50",
                            text: "text-gray-600",
                          },
                          "bg-orange-500": {
                            bg: "bg-orange-500",
                            border: "border-orange-400",
                            glow: "shadow-orange-500/50",
                            text: "text-orange-600",
                          },
                          "bg-green-500": {
                            bg: "bg-green-500",
                            border: "border-green-400",
                            glow: "shadow-green-500/50",
                            text: "text-green-600",
                          },
                          "bg-purple-500": {
                            bg: "bg-purple-500",
                            border: "border-purple-400",
                            glow: "shadow-purple-500/50",
                            text: "text-purple-600",
                          },
                        };
                        const colors =
                          colorMap[section.config.color] ||
                          colorMap["bg-gray-500"];

                        return (
                          <div
                            key={section.sectionKey}
                            className="flex items-center shrink-0"
                            ref={(el) => {
                              if (el) {
                                activeStepRefs.current.set(
                                  section.sectionKey,
                                  el
                                );
                              }
                            }}
                          >
                            {/* 步骤节点 */}
                            <button
                              onClick={() =>
                                scrollToSection(section.sectionKey)
                              }
                              className={`group relative flex flex-col items-center gap-1.5 px-2 py-1.5 rounded-lg transition-all duration-300 ${isActive
                                ? "scale-105"
                                : "hover:scale-102 hover:bg-gray-50 dark:hover:bg-gray-900/50"
                                }`}
                            >
                              {/* 圆圈容器 */}
                              <div className="relative">
                                {/* 脉动动画背景 */}
                                {isActive && (
                                  <div
                                    className={`absolute inset-0 ${colors.bg} rounded-full animate-ping opacity-20`}
                                  />
                                )}

                                {/* 主圆圈 */}
                                <div
                                  className={`relative w-9 h-9 rounded-full flex items-center justify-center font-semibold text-base transition-all duration-500 ${isActive
                                    ? `${colors.bg} text-white shadow-lg ${colors.glow
                                    } ring-2 ring-offset-1 ${colors.border.replace(
                                      "border-",
                                      "ring-"
                                    )} ring-opacity-30 dark:ring-offset-gray-950`
                                    : isCompleted
                                      ? "bg-gradient-to-br from-green-400 to-green-600 text-white shadow-md shadow-green-500/30"
                                      : "bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 border-2 border-gray-300 dark:border-gray-600 text-gray-400 dark:text-gray-500"
                                    } ${!isActive &&
                                    !isCompleted &&
                                    "group-hover:border-gray-400 dark:group-hover:border-gray-500 group-hover:shadow-md"
                                    }`}
                                >
                                  {/* 内容 */}
                                  {isCompleted ? (
                                    <Check className="w-4 h-4 animate-in zoom-in duration-300" />
                                  ) : (
                                    <span
                                      className={`text-base transition-transform duration-300 ${isActive
                                        ? "scale-110"
                                        : "group-hover:scale-105"
                                        }`}
                                    >
                                      {section.config.icon}
                                    </span>
                                  )}

                                  {/* 进度指示小点 */}
                                  {isActive && (
                                    <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-white dark:bg-gray-950 rounded-full flex items-center justify-center">
                                      <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* 标签 */}
                              <div
                                className={`text-[11px] font-semibold whitespace-nowrap transition-all duration-300 ${isActive
                                  ? `${colors.text} dark:text-white scale-105`
                                  : isCompleted
                                    ? "text-green-600 dark:text-green-400"
                                    : "text-gray-500 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-300"
                                  }`}
                              >
                                {section.type}
                              </div>

                              {/* 序号 */}
                              <div
                                className={`absolute top-0 left-0 w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold transition-all duration-300 ${isActive
                                  ? `${colors.bg} text-white shadow-sm`
                                  : isCompleted
                                    ? "bg-green-500 text-white"
                                    : "bg-gray-300 dark:bg-gray-600 text-gray-600 dark:text-gray-300"
                                  }`}
                              >
                                {idx + 1}
                              </div>
                            </button>

                            {/* 连接线 */}
                            {idx < allSections.length - 1 && (
                              <div className="relative w-16 h-1 mx-1">
                                {/* 背景轨道 */}
                                <div className="absolute inset-0 bg-gray-200 dark:bg-gray-700 rounded-full" />

                                {/* 进度条 */}
                                <div
                                  className={`absolute inset-0 rounded-full transition-all duration-700 ${isCompleted || isActive
                                    ? "bg-gradient-to-r from-green-400 to-green-500 shadow-sm shadow-green-500/30"
                                    : "bg-transparent"
                                    }`}
                                  style={{
                                    transform: isActive
                                      ? "scaleX(0.5)"
                                      : "scaleX(1)",
                                    transformOrigin: "left",
                                  }}
                                />

                                {/* 流动动画 */}
                                {isActive && (
                                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-50 animate-shimmer" />
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}

              {/* Chat Messages */}
              <div
                ref={messagesContainerRef}
                onScroll={(e) => {
                  const target = e.currentTarget;
                  const isBottom =
                    Math.abs(
                      target.scrollHeight - target.scrollTop - target.clientHeight
                    ) < 50;
                  stickToBottomRef.current = isBottom;
                }}
                className="flex-1 min-h-0 min-w-0 overflow-y-scroll overflow-x-hidden px-4 py-4 pr-5 space-y-6 scrollbar-auto"
              >
                {messages.map((message, msgIdx) => (
                  <ChatMessageItem
                    key={message.id}
                    message={message}
                    messageIndex={msgIdx}
                    isStreaming={
                      message.sender === "ai" && message.id === streamingMessageId
                    }
                    renderAssistant={renderMessageWithSections}
                    renderAssistantStreaming={renderMessageWithSectionsStreaming}
                  />
                ))}
                {/* 加载气泡已移除，改为仅按钮态提示 */}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-gray-200 dark:border-gray-800 shrink-0">
                <div className="flex gap-3 items-end">
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    onChange={handleFileUpload}
                    className="hidden"
                    accept="*"
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  >
                    <Paperclip className="h-4 w-4" />
                  </Button>
                  <div className="flex-1 relative">
                    <Input
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      placeholder="Ask anything..."
                      onKeyPress={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage();
                        }
                      }}
                      className="border-gray-200 dark:border-gray-700 bg-white dark:bg-black rounded-lg"
                    />
                  </div>
                  {/* 将清空按钮移动到发送按钮旁边 */}
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        title="清空聊天"
                        className="h-9 px-2"
                        disabled={isTyping}
                      >
                        <Eraser className="h-4 w-4" />
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>清空聊天？</AlertDialogTitle>
                        <AlertDialogDescription>
                          将删除当前会话内的所有消息，仅保留欢迎提示。
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>取消</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={clearChat}
                          className="bg-red-600 hover:bg-red-700"
                        >
                          确认清空
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                  {isTyping ? (
                    <Button
                      size="sm"
                      className="h-9 w-9 p-0 rounded-full bg-white text-black border border-blue-400/50 dark:bg-white dark:text-black"
                      title="正在生成…"
                      disabled
                    >
                      <RefreshCw className="h-4 w-4 animate-spin" />
                    </Button>
                  ) : (
                    <Button
                      onClick={handleSendMessage}
                      size="sm"
                      disabled={!inputValue.trim()}
                      className="bg-black text-white dark:bg-white dark:text-black hover:bg-gray-800 dark:hover:bg-gray-200"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Right Panel - Code Editor */}
          <ResizablePanel defaultSize={35} minSize={20}>
            <div className="flex flex-col bg-gray-50 dark:bg-gray-900 min-h-0 h-full">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800 h-12 shrink-0">
                <h2 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Code
                </h2>
                {showCodeEditor && (
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setShowCodeEditor(false);
                        setCodeEditorContent("");
                        setSelectedCodeSection("");
                      }}
                      className="h-6 px-2 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    >
                      Close
                    </Button>
                    <Button
                      size="sm"
                      onClick={executeCode}
                      disabled={!codeEditorContent || isExecutingCode}
                      className="h-6 px-3 text-xs bg-black text-white dark:bg-white dark:text-black"
                    >
                      {isExecutingCode ? "Running..." : "Run"}
                    </Button>
                  </div>
                )}
              </div>

              {!showCodeEditor ? (
                <div className="flex-1 flex items-center justify-center text-gray-400">
                  <div className="text-center select-none">
                    <p className="text-sm">Click a code block to edit</p>
                  </div>
                </div>
              ) : (
                <div className="flex-1 min-h-0 flex flex-col p-4 editor-container overflow-hidden">
                  {/* Code Editor */}
                  <div
                    className="min-h-0 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-black flex flex-col"
                    style={{ height: `${editorHeight}%` }}
                  >
                    <div className="bg-gray-50 dark:bg-gray-800 px-3 py-2 border-b border-gray-200 dark:border-gray-700 shrink-0">
                      <span className="text-xs text-gray-500 font-mono">
                        python
                      </span>
                    </div>
                    <div className="flex-1 min-h-0">
                      <Editor
                        height="100%"
                        defaultLanguage="python"
                        value={codeEditorContent}
                        onChange={(value) => setCodeEditorContent(value || "")}
                        theme={isDarkMode ? "vs-dark" : "light"}
                        options={{
                          fontSize: 14,
                          fontFamily:
                            "var(--font-mono), 'Courier New', monospace",
                          lineNumbers: "on",
                          minimap: { enabled: false },
                          scrollBeyondLastLine: false,
                          automaticLayout: true,
                          tabSize: 4,
                          insertSpaces: true,
                          wordWrap: "on",
                          folding: true,
                          lineDecorationsWidth: 10,
                          lineNumbersMinChars: 3,
                          glyphMargin: false,
                          selectOnLineNumbers: true,
                          roundedSelection: false,
                          readOnly: false,
                          cursorStyle: "line",
                          smoothScrolling: true,
                          formatOnPaste: true,
                          formatOnType: true,
                          suggestOnTriggerCharacters: true,
                          acceptSuggestionOnEnter: "on",
                          tabCompletion: "on",
                          scrollbar: {
                            vertical: "visible",
                            verticalScrollbarSize: 10,
                          },
                        }}
                        loading={
                          <div className="flex items-center justify-center h-full">
                            <div className="flex items-center gap-2 text-muted-foreground">
                              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                              <span className="text-sm">加载编辑器...</span>
                            </div>
                          </div>
                        }
                      />
                    </div>
                  </div>

                  {/* Resizer */}
                  <div
                    className="h-2 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 cursor-row-resize flex items-center justify-center group"
                    onMouseDown={handleMouseDown}
                  >
                    <div className="w-8 h-1 bg-gray-300 dark:bg-gray-600 rounded group-hover:bg-gray-400 dark:group-hover:bg-gray-500"></div>
                  </div>

                  {/* Terminal Output */}
                  <div
                    className="min-h-0 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-900 flex flex-col"
                    style={{ height: `${100 - editorHeight}%` }}
                  >
                    <div className="bg-gray-50 dark:bg-gray-800 px-3 py-2 border-b border-gray-200 dark:border-gray-700 shrink-0">
                      <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                        Output
                      </span>
                    </div>
                    <div className="flex-1 min-h-0 p-3 overflow-auto font-mono text-sm bg-white dark:bg-black text-gray-800 dark:text-gray-200">
                      {codeExecutionResult ? (
                        <div>
                          <div className="text-gray-500 dark:text-gray-400 mb-1">
                            $ python main.py
                          </div>
                          <pre className="whitespace-pre-wrap text-gray-800 dark:text-gray-200">
                            {codeExecutionResult}
                          </pre>
                          <div className="flex items-center mt-2">
                            <span className="text-gray-500 dark:text-gray-400">
                              $
                            </span>
                            <span className="w-2 h-4 bg-gray-400 dark:bg-gray-500 ml-1 animate-pulse"></span>
                          </div>
                        </div>
                      ) : (
                        <div className="text-gray-400 dark:text-gray-500 italic">
                          Run code to see output...
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
      {contextPos && contextTarget && (
        <div
          className="fixed z-50 bg-card border border-gray-200 dark:border-gray-700 rounded shadow-sm text-sm"
          style={{ left: contextPos.x, top: contextPos.y, minWidth: 180 }}
          onMouseLeave={closeContext}
        >
          {/* 生成文件专属：移动到普通文件区 */}
          {!contextTarget.is_dir &&
            contextTarget.path.startsWith("generated/") && (
              <button
                className="block w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-800"
                onClick={async () => {
                  await moveToDir(contextTarget.path, "");
                  closeContext();
                }}
              >
                移动到普通文件区
              </button>
            )}
          {!contextTarget.is_dir && (
            <button
              className="block w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-800"
              onClick={() => {
                openNode(contextTarget);
                closeContext();
              }}
            >
              预览
            </button>
          )}
          {!contextTarget.is_dir && contextTarget.download_url && (
            <a
              className="block px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-800"
              href={contextTarget.download_url}
              download={contextTarget.name}
              onClick={closeContext}
            >
              下载
            </a>
          )}
          <button
            className="block w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-800"
            onClick={() => {
              copyToClipboard(contextTarget.path)
                .then((ok) =>
                  toast({
                    description: ok ? "已复制路径" : "复制失败",
                    variant: ok ? undefined : "destructive",
                  })
                )
                .catch(() =>
                  toast({ description: "复制失败", variant: "destructive" })
                );
              closeContext();
            }}
          >
            复制路径
          </button>
          {!contextTarget.is_dir && (
            <button
              className="block w-full text-left px-3 py-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-950/20"
              onClick={() => {
                setDeleteConfirmPath(contextTarget.path);
                setDeleteIsDir(false);
              }}
            >
              删除文件
            </button>
          )}
          {contextTarget.is_dir && contextTarget.name === "generated" && (
            <button
              className="block w-full text-left px-3 py-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-950/20"
              onClick={() => {
                setDeleteConfirmPath(contextTarget.path);
                setDeleteIsDir(true);
              }}
            >
              删除文件夹
            </button>
          )}
        </div>
      )}
      {/* 全局删除确认弹窗 */}
      {/* 右键移动操作已集成到主菜单顶部，移除单独浮层 */}

      {/* 全局删除确认弹窗 */}
      <AlertDialog
        open={!!deleteConfirmPath}
        onOpenChange={(o) => !o && setDeleteConfirmPath(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteIsDir ? "确认删除文件夹？" : "确认删除文件？"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {deleteIsDir
                ? "此操作不可撤销，将删除该文件夹及其所有内容。"
                : "此操作不可撤销，将从 workspace 中移除此文件。"}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteConfirmPath(null)}>
              取消
            </AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={async () => {
                if (deleteConfirmPath) {
                  if (deleteIsDir) {
                    await deleteDir(deleteConfirmPath);
                  } else {
                    await deleteFile(deleteConfirmPath);
                  }
                }
                setDeleteConfirmPath(null);
                closeContext();
              }}
            >
              确认删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>


      {/* 文件预览弹窗 */}
      <Dialog open={isPreviewOpen} onOpenChange={setIsPreviewOpen}>
        <DialogContent
          style={{
            width: "90vw",
            height: "90vh",
            maxWidth: "90vw",
            maxHeight: "90vh",
          }}
          className=" p-0 overflow-hidden flex flex-col"
        >
          <DialogHeader className="px-4 py-3 border-b border-gray-200 dark:border-gray-800">
            <DialogTitle className="text-sm font-medium truncate">
              {previewTitle}
            </DialogTitle>
          </DialogHeader>
          <div
            ref={previewScrollRef}
            className="w-full flex-1 min-h-0 overflow-auto"
          >
            {previewLoading ? (
              <div className="h-full flex items-center justify-center text-sm text-gray-500">
                Loading...
              </div>
            ) : previewType === "image" ? (
              <div className="p-4 h-full flex items-center justify-center">
                <img
                  src={previewContent}
                  alt={previewTitle}
                  className="max-w-full max-h-full object-contain"
                />
              </div>
            ) : previewType === "pdf" ? (
              <iframe src={previewContent} className="w-full h-full" />
            ) : previewType === "text" ? (
              <div className="h-full min-h-0 p-2">
                <div className="h-full min-h-0 border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden">
                  <div className="h-full min-h-0">
                    <Editor
                      height="100%"
                      defaultLanguage={guessLanguageByExtension(
                        previewTitle.split(".").pop() || "text"
                      )}
                      language={guessLanguageByExtension(
                        previewTitle.split(".").pop() || "text"
                      )}
                      value={previewContent}
                      theme={isDarkMode ? "vs-dark" : "light"}
                      options={{
                        readOnly: true,
                        wordWrap: "on",
                        minimap: { enabled: false },
                        scrollBeyondLastLine: false,
                        fontFamily:
                          "var(--font-mono), 'Courier New', monospace",
                        fontSize: 14,
                        lineNumbers: "on",
                        automaticLayout: true,
                      }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4">
                <div className="text-xs text-gray-500 mb-2">
                  无法识别类型，尝试以文本方式预览：
                </div>
                <div className="border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden">
                  <SyntaxHighlighter
                    language={guessLanguageByExtension(
                      previewTitle.split(".").pop() || "text"
                    )}
                    style={isDarkMode ? oneDark : oneLight}
                    customStyle={{ margin: 0 }}
                    codeTagProps={{
                      style: {
                        fontFamily: "var(--font-mono)",
                        fontSize: "0.875rem",
                      },
                    }}
                  >
                    {previewContent}
                  </SyntaxHighlighter>
                </div>
                <div className="mt-3 text-xs text-gray-500">
                  如显示异常，
                  <a
                    className="underline"
                    href={previewDownloadUrl || previewContent}
                    target="_blank"
                    rel="noreferrer"
                  >
                    点击下载/打开
                  </a>
                </div>
              </div>
            )}
          </div>
          <div className="absolute bottom-4 right-4">
            <Button onClick={handleDownload} size="sm" variant="outline">
              <Download className="h-4 w-4" />
              下载
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
