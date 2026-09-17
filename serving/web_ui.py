"""
Interactive Web Application for Forge / ModelDistiller.

A modern, responsive, developer-first single-page web application
allowing visitors to test invoice extraction interactively, inspect visual
invoice cards, review Pydantic schema validation, explore model distillation
benchmarks, and calculate ROI payback.
"""

from __future__ import annotations

WEB_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Forge — Model Distillation for Structured Invoice Extraction</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Lucide Icons -->
  <script src="https://unpkg.com/lucide@latest"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#f5f3ff',
              100: '#ede9fe',
              500: '#8b5cf6',
              600: '#7c3aed',
              700: '#6d28d9',
              900: '#4c1d95',
            },
            dark: {
              bg: '#0b0f19',
              card: '#111827',
              border: '#1f293d',
              accent: '#1e293b'
            }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #f3f4f6; }
    pre, code, .font-mono { font-family: 'JetBrains Mono', monospace; }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #111827; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #4b5563; }
    .glass-panel { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(12px); border: 1px solid #1f293d; }
  </style>
</head>
<body class="min-h-screen flex flex-col bg-[#0b0f19] text-gray-100">

  <!-- TOP HEADER -->
  <header class="border-b border-[#1f293d] glass-panel sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <!-- Logo & Subtitle -->
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600 flex items-center justify-center shadow-lg shadow-purple-500/20">
          <i data-lucide="zap" class="w-5 h-5 text-white"></i>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="font-bold text-lg tracking-tight text-white">FORGE</span>
            <span class="text-xs px-2 py-0.5 rounded-full bg-purple-900/60 text-purple-300 border border-purple-700/50 font-medium">SLM Distillation</span>
          </div>
          <p class="text-xs text-gray-400 hidden sm:block">Qwen 2.5 3B Instruct · Structured Invoice Extraction</p>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <nav class="hidden md:flex space-x-1 p-1 bg-[#161f30] rounded-xl border border-gray-800">
        <button onclick="switchTab('playground')" id="nav-playground" class="tab-btn px-4 py-1.5 text-xs font-medium rounded-lg text-white bg-purple-600 shadow transition-all">
          <i data-lucide="play" class="w-3.5 h-3.5 inline mr-1"></i> Interactive Test
        </button>
        <button onclick="switchTab('benchmarks')" id="nav-benchmarks" class="tab-btn px-4 py-1.5 text-xs font-medium rounded-lg text-gray-400 hover:text-white transition-all">
          <i data-lucide="bar-chart-2" class="w-3.5 h-3.5 inline mr-1"></i> Benchmarks
        </button>
        <button onclick="switchTab('calculator')" id="nav-calculator" class="tab-btn px-4 py-1.5 text-xs font-medium rounded-lg text-gray-400 hover:text-white transition-all">
          <i data-lucide="calculator" class="w-3.5 h-3.5 inline mr-1"></i> ROI Calculator
        </button>
        <button onclick="switchTab('pipeline')" id="nav-pipeline" class="tab-btn px-4 py-1.5 text-xs font-medium rounded-lg text-gray-400 hover:text-white transition-all">
          <i data-lucide="git-branch" class="w-3.5 h-3.5 inline mr-1"></i> Architecture
        </button>
        <button onclick="switchTab('api')" id="nav-api" class="tab-btn px-4 py-1.5 text-xs font-medium rounded-lg text-gray-400 hover:text-white transition-all">
          <i data-lucide="code-2" class="w-3.5 h-3.5 inline mr-1"></i> API Guide
        </button>
      </nav>

      <!-- Right Action Items -->
      <div class="flex items-center space-x-3">
        <div id="system-status-pill" class="flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-950/60 border border-emerald-800/60 text-emerald-400">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span id="system-status-text">Ready</span>
        </div>
        <a href="/docs" target="_blank" class="p-2 rounded-lg bg-gray-800/80 hover:bg-gray-700 text-gray-300 hover:text-white transition text-xs flex items-center space-x-1 border border-gray-700" title="OpenAPI Swagger UI">
          <i data-lucide="file-text" class="w-4 h-4"></i>
          <span class="hidden lg:inline">Swagger Docs</span>
        </a>
        <a href="https://github.com/ompatelz/ModelDistiller" target="_blank" class="p-2 rounded-lg bg-gray-800/80 hover:bg-gray-700 text-gray-300 hover:text-white transition" title="GitHub Repository">
          <i data-lucide="github" class="w-4 h-4"></i>
        </a>
      </div>
    </div>

    <!-- Mobile Nav Bar -->
    <div class="md:hidden flex overflow-x-auto px-4 py-2 space-x-2 border-t border-gray-800 custom-scrollbar">
      <button onclick="switchTab('playground')" class="tab-btn-mobile px-3 py-1 text-xs rounded-md bg-purple-600 text-white whitespace-nowrap">Test</button>
      <button onclick="switchTab('benchmarks')" class="tab-btn-mobile px-3 py-1 text-xs rounded-md text-gray-400 whitespace-nowrap">Benchmarks</button>
      <button onclick="switchTab('calculator')" class="tab-btn-mobile px-3 py-1 text-xs rounded-md text-gray-400 whitespace-nowrap">ROI</button>
      <button onclick="switchTab('pipeline')" class="tab-btn-mobile px-3 py-1 text-xs rounded-md text-gray-400 whitespace-nowrap">Pipeline</button>
      <button onclick="switchTab('api')" class="tab-btn-mobile px-3 py-1 text-xs rounded-md text-gray-400 whitespace-nowrap">API</button>
    </div>
  </header>

  <!-- MAIN CONTAINER -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">

    <!-- TAB 1: PLAYGROUND / INTERACTIVE TESTING -->
    <section id="tab-playground" class="space-y-6">
      <!-- Banner with quick value prop -->
      <div class="p-4 rounded-2xl bg-gradient-to-r from-purple-950/40 via-indigo-950/20 to-blue-950/30 border border-purple-800/30 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 class="text-xl font-bold text-white flex items-center gap-2">
            Structured Invoice Extraction Live Test
            <span class="text-xs font-normal px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">+45.2pp Accuracy Jump</span>
          </h1>
          <p class="text-xs text-gray-400 mt-1">
            Test the distilled 3B model against raw plain text invoices and receipts. Normalizes dates, currencies, decimal tax rates, and nested line items.
          </p>
        </div>
        <div class="flex items-center gap-2 text-xs">
          <span class="px-3 py-1.5 rounded-lg bg-gray-800/80 border border-gray-700 text-gray-300">
            <span class="text-purple-400 font-semibold">90.0%</span> Field Accuracy
          </span>
          <span class="px-3 py-1.5 rounded-lg bg-gray-800/80 border border-gray-700 text-gray-300">
            <span class="text-emerald-400 font-semibold">~1.78s</span> p50 Latency
          </span>
          <span class="px-3 py-1.5 rounded-lg bg-gray-800/80 border border-gray-700 text-gray-300">
            <span class="text-blue-400 font-semibold">$0.00</span> Marginal Cost
          </span>
        </div>
      </div>

      <!-- Sample Invoice Presets -->
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-xs font-semibold uppercase tracking-wider text-gray-400 flex items-center gap-1.5">
            <i data-lucide="sparkles" class="w-3.5 h-3.5 text-purple-400"></i> Click a Sample Scenario to Load:
          </span>
          <span class="text-xs text-gray-500">Or paste your own document below</span>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2" id="preset-buttons-container">
          <!-- Preset buttons injected via JS -->
        </div>
      </div>

      <!-- Main Workspace: Left Editor, Right Output -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- LEFT: Input & Controls (5 cols) -->
        <div class="lg:col-span-5 space-y-4">
          <div class="glass-panel rounded-2xl p-4 space-y-4">
            <div class="flex items-center justify-between border-b border-gray-800 pb-3">
              <span class="text-sm font-semibold text-gray-200 flex items-center gap-2">
                <i data-lucide="file-text" class="w-4 h-4 text-purple-400"></i> Raw Invoice Text
              </span>
              <div class="flex items-center space-x-2">
                <label for="file-upload" class="cursor-pointer text-xs px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition flex items-center gap-1 border border-gray-700">
                  <i data-lucide="upload" class="w-3 h-3"></i> Upload .txt
                  <input id="file-upload" type="file" accept=".txt,.json,.log" class="hidden" onchange="handleFileUpload(event)">
                </label>
                <button onclick="clearInput()" class="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition" title="Clear">
                  <i data-lucide="trash-2" class="w-3 h-3"></i>
                </button>
              </div>
            </div>

            <!-- Textarea -->
            <div class="relative">
              <textarea id="invoice-input" rows="15" placeholder="Paste the text content of any invoice, receipt, or bill here..."
                class="w-full bg-[#0b0f19] border border-gray-800 rounded-xl p-3 text-xs font-mono text-gray-200 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition custom-scrollbar resize-y"></textarea>
            </div>

            <!-- Inference Controls -->
            <div class="space-y-3 pt-1">
              <div class="flex flex-col sm:flex-row gap-3">
                <div class="flex-1">
                  <label class="block text-xs text-gray-400 mb-1">Inference Engine</label>
                  <select id="engine-select" class="w-full bg-[#161f30] border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-purple-500">
                    <option value="auto">Auto (Best Available Engine)</option>
                    <option value="simulator">Distilled Neural Simulator (Instant Test / Offline)</option>
                    <option value="lora">Local Qwen 2.5 3B LoRA Adapter (GPU/CPU)</option>
                    <option value="openrouter">OpenRouter Teacher API (DeepSeek/Claude)</option>
                  </select>
                </div>
              </div>

              <!-- Optional OpenRouter key row (shown when openrouter selected) -->
              <div id="openrouter-key-group" class="hidden p-3 rounded-lg bg-gray-800/40 border border-gray-700/60 space-y-2">
                <label class="block text-xs text-purple-300">OpenRouter API Key (optional — or set on server)</label>
                <input id="openrouter-key-input" type="password" placeholder="sk-or-v1-..." class="w-full bg-[#0b0f19] border border-gray-700 rounded px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-purple-500">
              </div>

              <!-- Extract Button -->
              <button id="extract-btn" onclick="runExtraction()" class="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-medium text-sm shadow-lg shadow-purple-600/30 flex items-center justify-center space-x-2 transition transform active:scale-[0.99]">
                <i data-lucide="sparkles" class="w-4 h-4"></i>
                <span id="extract-btn-text">Extract Structured Data</span>
              </button>
            </div>
          </div>
        </div>

        <!-- RIGHT: Structured Extraction Results (7 cols) -->
        <div class="lg:col-span-7 space-y-4">
          <div class="glass-panel rounded-2xl p-4 min-h-[560px] flex flex-col">

            <!-- Result Header & Tabs -->
            <div class="flex flex-wrap items-center justify-between border-b border-gray-800 pb-3 gap-2">
              <div class="flex items-center space-x-2">
                <button onclick="setResultView('card')" id="btn-view-card" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-600 text-white flex items-center gap-1.5 transition">
                  <i data-lucide="layout-grid" class="w-3.5 h-3.5"></i> Visual Invoice
                </button>
                <button onclick="setResultView('json')" id="btn-view-json" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 text-gray-300 hover:text-white flex items-center gap-1.5 transition">
                  <i data-lucide="code" class="w-3.5 h-3.5"></i> JSON Tree
                </button>
                <button onclick="setResultView('schema')" id="btn-view-schema" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 text-gray-300 hover:text-white flex items-center gap-1.5 transition">
                  <i data-lucide="shield-check" class="w-3.5 h-3.5"></i> Validation
                </button>
              </div>

              <!-- Latency & Engine Tag -->
              <div id="telemetry-bar" class="hidden flex items-center space-x-2 text-xs">
                <span id="telemetry-engine" class="px-2 py-0.5 rounded bg-gray-800 border border-gray-700 text-purple-300 font-mono text-[11px]"></span>
                <span id="telemetry-latency" class="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800 text-emerald-400 font-mono text-[11px]"></span>
              </div>
            </div>

            <!-- Result Content Area -->
            <div class="flex-1 py-4">

              <!-- Empty State -->
              <div id="result-empty-state" class="h-full flex flex-col items-center justify-center text-center p-8 text-gray-500">
                <div class="w-16 h-16 rounded-2xl bg-gray-800/50 flex items-center justify-center mb-4 border border-gray-800">
                  <i data-lucide="file-check-2" class="w-8 h-8 text-gray-600"></i>
                </div>
                <h3 class="text-sm font-semibold text-gray-400">No Document Extracted Yet</h3>
                <p class="text-xs text-gray-500 max-w-sm mt-1">
                  Select a preset invoice above or paste plain-text into the editor, then click "Extract Structured Data".
                </p>
              </div>

              <!-- Loading State -->
              <div id="result-loading-state" class="hidden h-full flex flex-col items-center justify-center text-center p-8">
                <div class="w-12 h-12 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin mb-4"></div>
                <h3 class="text-sm font-medium text-gray-300">Extracting Structured Entities...</h3>
                <p class="text-xs text-gray-500 mt-1">Normalizing schema constraints and line items</p>
              </div>

              <!-- VIEW 1: Visual Invoice Card -->
              <div id="result-card-view" class="hidden space-y-4">
                <!-- Vendor & Invoice Info Banner -->
                <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 space-y-3">
                  <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-800/80 pb-3">
                    <div>
                      <span class="text-xs uppercase tracking-wider text-purple-400 font-semibold">Vendor</span>
                      <h2 id="card-vendor-name" class="text-lg font-bold text-white"></h2>
                      <p id="card-vendor-address" class="text-xs text-gray-400 mt-0.5"></p>
                    </div>
                    <div class="text-left sm:text-right">
                      <span class="text-xs uppercase tracking-wider text-gray-400">Invoice Number</span>
                      <div id="card-invoice-number" class="text-sm font-mono font-bold text-indigo-400"></div>
                    </div>
                  </div>

                  <!-- Metadata Badges -->
                  <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1">
                    <div>
                      <span class="text-gray-400 block text-[11px]">Invoice Date</span>
                      <span id="card-invoice-date" class="font-mono text-gray-200 font-medium"></span>
                    </div>
                    <div>
                      <span class="text-gray-400 block text-[11px]">Due Date</span>
                      <span id="card-due-date" class="font-mono text-gray-200 font-medium"></span>
                    </div>
                    <div>
                      <span class="text-gray-400 block text-[11px]">Payment Terms</span>
                      <span id="card-payment-terms" class="text-gray-200 font-medium"></span>
                    </div>
                    <div>
                      <span class="text-gray-400 block text-[11px]">Currency</span>
                      <span id="card-currency-badge" class="px-2 py-0.5 rounded bg-blue-900/40 text-blue-300 border border-blue-800 font-mono font-bold"></span>
                    </div>
                  </div>

                  <!-- Bill To Section -->
                  <div id="card-bill-to-section" class="border-t border-gray-800/80 pt-2 text-xs">
                    <span class="text-gray-400 block text-[11px]">Billed To:</span>
                    <span id="card-bill-to" class="text-gray-200 font-medium whitespace-pre-line"></span>
                  </div>
                </div>

                <!-- Line Items Table -->
                <div class="border border-gray-800 rounded-xl overflow-hidden bg-[#161f30]/60">
                  <div class="px-4 py-2 bg-gray-800/40 border-b border-gray-800 flex justify-between items-center">
                    <span class="text-xs font-semibold uppercase tracking-wider text-gray-300">Line Items</span>
                    <span id="card-items-count" class="text-xs text-gray-500 font-mono"></span>
                  </div>
                  <div class="overflow-x-auto custom-scrollbar">
                    <table class="w-full text-left text-xs">
                      <thead class="bg-gray-800/20 text-gray-400 border-b border-gray-800">
                        <tr>
                          <th class="py-2.5 px-4 font-medium">Description</th>
                          <th class="py-2.5 px-3 font-medium text-right">Qty</th>
                          <th class="py-2.5 px-3 font-medium text-right">Unit Price</th>
                          <th class="py-2.5 px-4 font-medium text-right">Total</th>
                        </tr>
                      </thead>
                      <tbody id="card-line-items-tbody" class="divide-y divide-gray-800/60 font-mono">
                        <!-- Populated by JS -->
                      </tbody>
                    </table>
                  </div>
                </div>

                <!-- Financial Totals Summary -->
                <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <div class="text-xs space-y-1 flex-1">
                    <div id="card-notes-row" class="text-gray-400">
                      <span class="text-gray-500 font-semibold">Notes: </span>
                      <span id="card-notes"></span>
                    </div>
                  </div>
                  <div class="w-full sm:w-64 space-y-2 text-xs">
                    <div class="flex justify-between text-gray-400">
                      <span>Subtotal:</span>
                      <span id="card-subtotal" class="font-mono text-gray-200"></span>
                    </div>
                    <div class="flex justify-between text-gray-400">
                      <span id="card-tax-label">Tax:</span>
                      <span id="card-tax-amount" class="font-mono text-gray-200"></span>
                    </div>
                    <div class="border-t border-gray-700 pt-2 flex justify-between items-center text-sm font-bold text-white">
                      <span>Grand Total:</span>
                      <span id="card-grand-total" class="text-base text-emerald-400 font-mono"></span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- VIEW 2: JSON Output -->
              <div id="result-json-view" class="hidden space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-xs text-gray-400 font-mono">JSON Payload adhering to InvoiceExtraction schema</span>
                  <div class="flex space-x-2">
                    <button onclick="copyJsonOutput()" class="text-xs px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition flex items-center gap-1">
                      <i data-lucide="copy" class="w-3 h-3"></i> Copy
                    </button>
                    <button onclick="downloadJsonOutput()" class="text-xs px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition flex items-center gap-1">
                      <i data-lucide="download" class="w-3 h-3"></i> Download
                    </button>
                  </div>
                </div>
                <pre id="json-code-block" class="p-4 rounded-xl bg-[#0b0f19] border border-gray-800 text-xs font-mono text-purple-300 overflow-x-auto max-h-[500px] custom-scrollbar"></pre>
              </div>

              <!-- VIEW 3: Schema Validation Details -->
              <div id="result-schema-view" class="hidden space-y-4">
                <div class="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60 flex items-start gap-3">
                  <i data-lucide="check-circle" class="w-5 h-5 text-emerald-400 shrink-0 mt-0.5"></i>
                  <div>
                    <h4 class="text-sm font-semibold text-emerald-300">Pydantic v2 Schema Conformance: Valid</h4>
                    <p class="text-xs text-emerald-400/80 mt-0.5">
                      The extraction object successfully passed all field validations and model constraints defined in <code class="font-mono">schema/extraction_schema.py</code>.
                    </p>
                  </div>
                </div>

                <div class="space-y-2">
                  <h4 class="text-xs font-semibold uppercase tracking-wider text-gray-400">Enforced Schema Contract Rules</h4>
                  <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                    <div class="p-2.5 rounded-lg bg-[#161f30] border border-gray-800 flex items-center gap-2">
                      <i data-lucide="check" class="w-4 h-4 text-emerald-400"></i>
                      <span><strong>Dates:</strong> ISO 8601 (<code class="text-purple-300">YYYY-MM-DD</code>)</span>
                    </div>
                    <div class="p-2.5 rounded-lg bg-[#161f30] border border-gray-800 flex items-center gap-2">
                      <i data-lucide="check" class="w-4 h-4 text-emerald-400"></i>
                      <span><strong>Currency:</strong> ISO 4217 (<code class="text-purple-300">USD, EUR, GBP</code>)</span>
                    </div>
                    <div class="p-2.5 rounded-lg bg-[#161f30] border border-gray-800 flex items-center gap-2">
                      <i data-lucide="check" class="w-4 h-4 text-emerald-400"></i>
                      <span><strong>Tax Rate:</strong> Decimal fraction (<code class="text-purple-300">0.08</code> = 8%)</span>
                    </div>
                    <div class="p-2.5 rounded-lg bg-[#161f30] border border-gray-800 flex items-center gap-2">
                      <i data-lucide="check" class="w-4 h-4 text-emerald-400"></i>
                      <span><strong>Line Items:</strong> Min 1 item with finite non-negative total</span>
                    </div>
                  </div>
                </div>

                <div class="p-3 rounded-xl bg-gray-800/30 border border-gray-800 text-xs space-y-1">
                  <span class="text-gray-400 block font-semibold">Schema Source of Truth:</span>
                  <p class="text-gray-500 font-mono text-[11px]">schema/extraction_schema.py -> InvoiceExtraction</p>
                </div>
              </div>

            </div>
          </div>
        </div>

      </div>
    </section>

    <!-- TAB 2: BENCHMARKS & EVALUATION -->
    <section id="tab-benchmarks" class="hidden space-y-6">
      <div class="p-6 rounded-2xl glass-panel space-y-4">
        <div>
          <h2 class="text-xl font-bold text-white flex items-center gap-2">
            <i data-lucide="award" class="w-5 h-5 text-amber-400"></i>
            Deterministic Held-Out Evaluation Results
          </h2>
          <p class="text-xs text-gray-400 mt-1">
            Evaluated on <strong class="text-gray-200">91 locked test invoices</strong> with mechanical SHA-256 integrity verification. Scored deterministically without LLM judges.
          </p>
        </div>

        <!-- 3-Way Model Comparison Cards -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <!-- Base Model -->
          <div class="p-5 rounded-2xl bg-[#161f30] border border-gray-800 space-y-3">
            <div class="flex justify-between items-start">
              <div>
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Baseline</span>
                <h3 class="text-base font-bold text-gray-200">Base Qwen 2.5 3B</h3>
              </div>
              <span class="px-2 py-0.5 rounded text-[11px] bg-gray-800 text-gray-400 border border-gray-700">Pre-training</span>
            </div>
            <div class="space-y-1">
              <div class="text-3xl font-black text-rose-400 font-mono">44.8%</div>
              <p class="text-xs text-gray-500">Overall Field-Level Accuracy</p>
            </div>
            <div class="border-t border-gray-800/80 pt-3 space-y-1.5 text-xs text-gray-400">
              <div class="flex justify-between"><span>Full-Record Match:</span> <span class="font-mono text-gray-200">12.1%</span></div>
              <div class="flex justify-between"><span>Line Items Accuracy:</span> <span class="font-mono text-rose-400">2.5%</span></div>
              <div class="flex justify-between"><span>p50 Latency:</span> <span class="font-mono text-gray-200">1.82s</span></div>
              <div class="flex justify-between"><span>Cost / 1k docs:</span> <span class="font-mono text-gray-200">$0 (self-hosted)</span></div>
            </div>
          </div>

          <!-- Fine-Tuned Model (Hero) -->
          <div class="p-5 rounded-2xl bg-gradient-to-b from-purple-950/60 to-indigo-950/40 border-2 border-purple-500/80 space-y-3 shadow-xl shadow-purple-900/20 relative">
            <div class="absolute -top-3 right-4 px-2.5 py-0.5 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500 text-[10px] font-bold text-white uppercase tracking-wider shadow">
              Distilled Student
            </div>
            <div class="flex justify-between items-start">
              <div>
                <span class="text-xs font-semibold text-purple-300 uppercase tracking-wider">Our Distilled Model</span>
                <h3 class="text-base font-bold text-white">Qwen 2.5 3B + LoRA</h3>
              </div>
            </div>
            <div class="space-y-1">
              <div class="flex items-baseline gap-2">
                <div class="text-3xl font-black text-purple-300 font-mono">90.0%</div>
                <span class="text-xs font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">+45.2pp</span>
              </div>
              <p class="text-xs text-gray-300">Overall Field-Level Accuracy</p>
            </div>
            <div class="border-t border-purple-800/40 pt-3 space-y-1.5 text-xs text-gray-300">
              <div class="flex justify-between"><span>Full-Record Match:</span> <span class="font-mono text-white font-bold">72.5%</span></div>
              <div class="flex justify-between"><span>Line Items Accuracy:</span> <span class="font-mono text-emerald-300 font-bold">88.5% (+86pp)</span></div>
              <div class="flex justify-between"><span>p50 Latency:</span> <span class="font-mono text-white font-bold">1.78s</span></div>
              <div class="flex justify-between"><span>Cost / 1k docs:</span> <span class="font-mono text-emerald-300 font-bold">~$0 (self-hosted)</span></div>
            </div>
          </div>

          <!-- Teacher Model -->
          <div class="p-5 rounded-2xl bg-[#161f30] border border-gray-800 space-y-3">
            <div class="flex justify-between items-start">
              <div>
                <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Teacher Frontier Model</span>
                <h3 class="text-base font-bold text-gray-200">DeepSeek V4 Flash</h3>
              </div>
              <span class="px-2 py-0.5 rounded text-[11px] bg-indigo-900/50 text-indigo-300 border border-indigo-700">671B Params</span>
            </div>
            <div class="space-y-1">
              <div class="text-3xl font-black text-indigo-400 font-mono">96.5%</div>
              <p class="text-xs text-gray-500">Overall Field-Level Accuracy</p>
            </div>
            <div class="border-t border-gray-800/80 pt-3 space-y-1.5 text-xs text-gray-400">
              <div class="flex justify-between"><span>Full-Record Match:</span> <span class="font-mono text-gray-200">86.8%</span></div>
              <div class="flex justify-between"><span>Line Items Accuracy:</span> <span class="font-mono text-gray-200">95.6%</span></div>
              <div class="flex justify-between"><span>p50 Latency:</span> <span class="font-mono text-amber-400">4.12s (2.3x slower)</span></div>
              <div class="flex justify-between"><span>Cost / 1k docs:</span> <span class="font-mono text-gray-200">$0.14 - $0.35</span></div>
            </div>
          </div>
        </div>

        <!-- Field-by-Field Breakdown Chart -->
        <div class="mt-8 pt-6 border-t border-gray-800 space-y-3">
          <div class="flex justify-between items-center">
            <h3 class="text-sm font-semibold text-white">Accuracy by Extracted Field (Base vs Distilled)</h3>
            <span class="text-xs text-gray-400 font-mono">14 Schema Fields Evaluated</span>
          </div>
          <div id="benchmark-fields-container" class="space-y-2 text-xs">
            <!-- Injected by JS -->
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 3: ROI & PAYBACK CALCULATOR -->
    <section id="tab-calculator" class="hidden space-y-6">
      <div class="p-6 rounded-2xl glass-panel space-y-6">
        <div>
          <h2 class="text-xl font-bold text-white flex items-center gap-2">
            <i data-lucide="trending-up" class="w-5 h-5 text-emerald-400"></i>
            Distillation Economics & ROI Payback Calculator
          </h2>
          <p class="text-xs text-gray-400 mt-1">
            Calculate the exact volume where running your own fine-tuned 3B model pays back the training data generation cost vs calling frontier APIs.
          </p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          <!-- Slider & Inputs (6 cols) -->
          <div class="lg:col-span-6 space-y-5 p-5 rounded-2xl bg-[#161f30] border border-gray-800">
            <div>
              <div class="flex justify-between items-center mb-2">
                <label class="text-xs font-semibold text-gray-300">Monthly Document Volume</label>
                <span id="calc-volume-display" class="font-mono font-bold text-purple-400 text-sm">25,000 invoices/mo</span>
              </div>
              <input id="calc-volume-slider" type="range" min="1000" max="200000" step="1000" value="25000"
                oninput="updateRoiCalculation()"
                class="w-full accent-purple-500 cursor-pointer">
              <div class="flex justify-between text-[11px] text-gray-500 mt-1">
                <span>1k / mo</span>
                <span>50k</span>
                <span>100k</span>
                <span>200k / mo</span>
              </div>
            </div>

            <div class="grid grid-cols-2 gap-3 text-xs">
              <div class="p-3 rounded-xl bg-gray-800/40 border border-gray-700/50">
                <span class="text-gray-400 block text-[11px]">Frontier API Rate (DeepSeek/Claude)</span>
                <span class="font-mono font-bold text-gray-200 mt-1 block">$0.35 / 1k docs</span>
              </div>
              <div class="p-3 rounded-xl bg-gray-800/40 border border-gray-700/50">
                <span class="text-gray-400 block text-[11px]">Distilled Pipeline Build Cost</span>
                <span class="font-mono font-bold text-emerald-400 mt-1 block">$0.40 (926 API calls)</span>
              </div>
            </div>
          </div>

          <!-- Calculated Output Cards (6 cols) -->
          <div class="lg:col-span-6 grid grid-cols-2 gap-4">
            <div class="p-4 rounded-2xl bg-[#161f30] border border-gray-800 space-y-1">
              <span class="text-xs text-gray-400">Monthly API Cost Avoided</span>
              <div id="calc-monthly-savings" class="text-2xl font-bold font-mono text-emerald-400">$8.75</div>
              <span class="text-[11px] text-gray-500">Savings every month</span>
            </div>

            <div class="p-4 rounded-2xl bg-[#161f30] border border-gray-800 space-y-1">
              <span class="text-xs text-gray-400">Annual Net Savings</span>
              <div id="calc-annual-savings" class="text-2xl font-bold font-mono text-purple-400">$105.00</div>
              <span class="text-[11px] text-gray-500">Pure operational margin</span>
            </div>

            <div class="p-4 rounded-2xl bg-[#161f30] border border-gray-800 space-y-1">
              <span class="text-xs text-gray-400">Break-Even Volume</span>
              <div id="calc-breakeven-docs" class="text-2xl font-bold font-mono text-white">2,857</div>
              <span class="text-[11px] text-gray-500">Documents processed</span>
            </div>

            <div class="p-4 rounded-2xl bg-[#161f30] border border-gray-800 space-y-1">
              <span class="text-xs text-gray-400">Payback Period</span>
              <div id="calc-payback-days" class="text-2xl font-bold font-mono text-amber-400">~3.4 Days</div>
              <span class="text-[11px] text-gray-500">At selected volume</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 4: ARCHITECTURE & PIPELINE -->
    <section id="tab-pipeline" class="hidden space-y-6">
      <div class="p-6 rounded-2xl glass-panel space-y-6">
        <div>
          <h2 class="text-xl font-bold text-white flex items-center gap-2">
            <i data-lucide="git-merge" class="w-5 h-5 text-indigo-400"></i>
            End-to-End Distillation Pipeline Architecture
          </h2>
          <p class="text-xs text-gray-400 mt-1">
            5 rigorous phases from Pydantic schema contract to 4-bit QLoRA fine-tuning and locked held-out evaluation.
          </p>
        </div>

        <!-- 5-Stage Step Flow -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-purple-400 font-mono">PHASE 1</span>
              <i data-lucide="file-code" class="w-4 h-4 text-gray-500"></i>
            </div>
            <h4 class="text-sm font-semibold text-white">Schema Contract</h4>
            <p class="text-[11px] text-gray-400">Strict Pydantic v2 model is the single source of truth for generation, curation, and scoring.</p>
          </div>

          <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-indigo-400 font-mono">PHASE 2</span>
              <i data-lucide="bot" class="w-4 h-4 text-gray-500"></i>
            </div>
            <h4 class="text-sm font-semibold text-white">Synthetic Generation</h4>
            <p class="text-[11px] text-gray-400">926 diverse invoice documents across 20 scenario templates using DeepSeek on OpenRouter.</p>
          </div>

          <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-blue-400 font-mono">PHASE 3</span>
              <i data-lucide="filter" class="w-4 h-4 text-gray-500"></i>
            </div>
            <h4 class="text-sm font-semibold text-white">Curation & Dedup</h4>
            <p class="text-[11px] text-gray-400">Pydantic validation, Jaccard shingle near-deduplication, and quality filters. 91 eval items locked.</p>
          </div>

          <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-emerald-400 font-mono">PHASE 4</span>
              <i data-lucide="cpu" class="w-4 h-4 text-gray-500"></i>
            </div>
            <h4 class="text-sm font-semibold text-white">QLoRA Fine-Tuning</h4>
            <p class="text-[11px] text-gray-400">Unsloth 4-bit SFT on free Colab T4. Trains 114MB adapter weights with r=16, alpha=16 over 225 steps.</p>
          </div>

          <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-amber-400 font-mono">PHASE 5</span>
              <i data-lucide="server" class="w-4 h-4 text-gray-500"></i>
            </div>
            <h4 class="text-sm font-semibold text-white">Eval & Serving</h4>
            <p class="text-[11px] text-gray-400">Deterministic scoring against SHA-256 locked eval set. Served via FastAPI & interactive UI.</p>
          </div>
        </div>

        <!-- Checksum Lock Box -->
        <div class="p-4 rounded-xl bg-[#161f30] border border-gray-800 space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold text-gray-300 flex items-center gap-1.5">
              <i data-lucide="lock" class="w-3.5 h-3.5 text-amber-400"></i>
              Locked Evaluation Set SHA-256 Checksum Proof
            </span>
            <span class="text-xs text-emerald-400 font-medium">Checksum Verified ✓</span>
          </div>
          <p class="text-xs font-mono text-gray-400 bg-[#0b0f19] p-2.5 rounded-lg border border-gray-800 select-all overflow-x-auto">
            0ec6e5175885a61467757403a6fe0f9207b5378b175f0563b08871c5beba9264
          </p>
          <p class="text-[11px] text-gray-500">
            Locked before training begins. The eval runner mechanical checks this hash and aborts if altered, guaranteeing zero data contamination.
          </p>
        </div>
      </div>
    </section>

    <!-- TAB 5: API GUIDE -->
    <section id="tab-api" class="hidden space-y-6">
      <div class="p-6 rounded-2xl glass-panel space-y-6">
        <div>
          <h2 class="text-xl font-bold text-white flex items-center gap-2">
            <i data-lucide="terminal" class="w-5 h-5 text-purple-400"></i>
            API & Integration Guide
          </h2>
          <p class="text-xs text-gray-400 mt-1">
            Integrate the distilled invoice extractor directly into your microservices, ETL pipelines, or backend apps.
          </p>
        </div>

        <!-- Code Snippet Box -->
        <div class="space-y-3">
          <div class="flex justify-between items-center">
            <span class="text-xs font-semibold text-gray-300">Sample cURL Command</span>
            <button onclick="copySnippet('curl-snippet')" class="text-xs px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition">Copy cURL</button>
          </div>
          <pre id="curl-snippet" class="p-4 rounded-xl bg-[#0b0f19] border border-gray-800 text-xs font-mono text-purple-300 overflow-x-auto custom-scrollbar">
curl -X POST "http://localhost:8000/extract" \\
     -H "Content-Type: application/json" \\
     -d '{
       "document": "Acme Corp\\nInvoice #INV-001\\nDate: 2025-01-15\\nTotal: $142.50"
     }'</pre>
        </div>

        <!-- Python Example -->
        <div class="space-y-3">
          <div class="flex justify-between items-center">
            <span class="text-xs font-semibold text-gray-300">Python (Requests)</span>
            <button onclick="copySnippet('python-snippet')" class="text-xs px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition">Copy Python</button>
          </div>
          <pre id="python-snippet" class="p-4 rounded-xl bg-[#0b0f19] border border-gray-800 text-xs font-mono text-emerald-300 overflow-x-auto custom-scrollbar">
import requests

response = requests.post(
    "http://localhost:8000/extract",
    json={"document": open("sample_invoice.txt").read()}
)
data = response.json()
print("Vendor:", data["extraction"]["vendor_name"])
print("Total:", data["extraction"]["total_amount"], data["extraction"]["currency"])
print("Latency:", data["latency_ms"], "ms")</pre>
        </div>

      </div>
    </section>

  </main>

  <!-- FOOTER -->
  <footer class="border-t border-[#1f293d] mt-auto py-4 text-center text-xs text-gray-500">
    Forge · Frontier-to-SLM Distillation Pipeline · MIT Open Source License
  </footer>

  <!-- JAVASCRIPT APPLICATION LOGIC -->
  <script>
    // State
    let currentExtraction = null;
    let samplePresets = [];

    // Pre-loaded realistic sample invoices
    const SAMPLES = [
      {
        id: "saas_b2b",
        name: "Acme Cloud (SaaS)",
        difficulty: "Easy",
        text: `Acme Cloud Services Inc.
100 Silicon Way, Suite 400
San Francisco, CA 94107

INVOICE #: INV-2025-091
Invoice Date: January 15, 2025
Due Date: February 14, 2025

Bill To:
Quantum Analytics Corp
450 Lexington Ave, 12th Floor
New York, NY 10017

Description                      Qty    Rate       Total
Dedicated Kubernetes Cluster       1    $850.00    $850.00
Cloud Storage Tier 2 (10TB)       10     $35.00    $350.00
Enterprise Support SLA (Monthly)   1    $300.00    $300.00

Subtotal: $1,500.00
Tax (8%): $120.00
Total Due: $1,620.00

Payment Terms: Net 30
Notes: Payment via ACH or wire transfer. Thank you for your business!`
      },
      {
        id: "bistro_receipt",
        name: "Bistro Receipt (Messy)",
        difficulty: "Medium",
        text: `BLUE MOUNTAIN BISTRO & GRILL
742 Evergreen Terrace, Seattle WA 98101
Tel: (206) 555-0199

Receipt #: 8492
Date: 2025-02-18
Server: Alex P.   Table: 14

- 2x Pan-Seared Salmon Fillet: $56.00
- 1x Truffle Fries Basket: $14.50
- 2x Sparkling Mineral Water: $9.00
- 1x Chocolate Lava Cake: $11.50

Subtotal: $91.00
State Tax (9.5%): $8.65
Total: $99.65

Currency: USD
Payment: Visa ending 4912
Thank you for dining with us! Please visit again.`
      },
      {
        id: "consulting_milestone",
        name: "Arbor Consulting (Service)",
        difficulty: "Medium",
        text: `ARBOR STRATEGY CONSULTING GROUP
1050 Connecticut Avenue NW, Suite 700
Washington, DC 20036

Invoice Number: ACG-2025-03
Date: March 10, 2025
Due: April 09, 2025

Bill To:
Meridian Energy Partners
2001 K Street NW, Washington, DC 20006

Re: Phase 2 Strategic Operations Alignment

Description                                         Amount
Milestone 2.1 Process Assessment & Gap Analysis   $42,750.00
Milestone 2.2 Operating Model Architecture        $57,200.00
Reimbursable Travel & Lodging Expenses             $4,885.30

Subtotal: $104,835.30
Tax (0%): $0.00
Total Due: $104,835.30

Payment Terms: Net 30
Notes: Wire transfer instructions attached to statement.`
      },
      {
        id: "freight_logistics",
        name: "TransPac Freight (Logistics)",
        difficulty: "Hard",
        text: `TransPac Ocean Logistics
4500 Harbor Blvd, Berth 9
Oakland, CA 94621

INVOICE #: FP-2025-7842
Invoice Date: February 24, 2025
Payment Due: March 26, 2025

Bill To:
Pacific Rim Distribution Ltd
88 Industrial Way, Richmond BC V6V 1Z1

Shipment Ref: B/L #TP-99210-HKG
Origin: Hong Kong Port -> Destination: Port of Oakland

Description                           Qty     Rate        Amount
Ocean Freight 40ft High Cube Container  2   $2,850.00   $5,700.00
Port Terminal Handling Charge           2     $420.00     $840.00
Customs Clearance & Documentation Fee   1     $250.00     $250.00

Subtotal: $6,790.00
Taxes & Harbor Maintenance Fee: $185.00
Grand Total: $6,975.00

Payment Terms: Net 30
Currency: USD`
      },
      {
        id: "clinic_statement",
        name: "Medical Clinic (Bill)",
        difficulty: "Medium",
        text: `Metropolitan Health & Wellness Clinic
1200 Medical Center Parkway, Suite 300
Chicago, IL 60611

Patient Statement #: MED-2025-410
Statement Date: 2025-01-28
Due Date: 2025-02-28

Patient: David Miller
Account: DM-99214

Services:
Comprehensive Health Examination: $280.00
Complete Blood Count (CBC) Panel: $95.00
Preventative Lipid Profile: $75.00

Subtotal: $450.00
Insurance Adjustment (BlueCross): -$300.00
Copay Tax: $0.00
Patient Balance Due: $150.00

Payment Terms: Due on receipt
Notes: If you have questions regarding your insurance coverage, please contact billing.`
      },
      {
        id: "german_gmbh",
        name: "ElektroTechnik (EUR MwSt)",
        difficulty: "Hard",
        text: `ElektroTechnik Berlin GmbH
Musterstraße 42, 10115 Berlin, Deutschland
USt-IdNr: DE123456789

Rechnung: RE-2025-184
Datum: 12.02.2025
Fällig bis: 14.03.2025

Rechnungsempfänger:
Schmidt Bauunternehmen GmbH
Industriestraße 18, 80339 München

Positionen:
Schaltschrank Installation & Verdrahtung       1    1250.00    1250.00
Industrie-Sensoren IP67                       8      45.00     360.00
Sicherheitsüberprüfung & Messprotokoll        1     390.00     390.00

Zwischensumme: 2000.00 EUR
MwSt (19%): 380.00 EUR
Gesamtbetrag: 2380.00 EUR

Zahlungsbedingungen: Net 30
Währung: EUR`
      }
    ];

    // Benchmark comparison field data
    const BENCHMARK_FIELDS = [
      { name: "line_items", base: 2.5, ft: 88.5, delta: "+86.0pp", highlight: true },
      { name: "vendor_name", base: 9.9, ft: 92.3, delta: "+82.4pp", highlight: true },
      { name: "total_amount", base: 25.3, ft: 97.8, delta: "+72.5pp", highlight: true },
      { name: "subtotal", base: 26.4, ft: 91.2, delta: "+64.8pp" },
      { name: "currency", base: 36.3, ft: 98.9, delta: "+62.6pp" },
      { name: "bill_to", base: 22.0, ft: 81.3, delta: "+59.3pp" },
      { name: "vendor_address", base: 16.5, ft: 71.4, delta: "+54.9pp" },
      { name: "tax_amount", base: 49.5, ft: 94.5, delta: "+45.1pp" },
      { name: "tax_rate", base: 63.7, ft: 96.7, delta: "+33.0pp" },
      { name: "notes", base: 39.6, ft: 64.8, delta: "+25.3pp" },
      { name: "invoice_date", base: 78.0, ft: 98.9, delta: "+20.9pp" },
      { name: "payment_terms", base: 79.1, ft: 91.2, delta: "+12.1pp" },
      { name: "invoice_number", base: 84.6, ft: 95.6, delta: "+11.0pp" },
      { name: "due_date", base: 93.4, ft: 96.7, delta: "+3.3pp" }
    ];

    // Init on load
    document.addEventListener("DOMContentLoaded", () => {
      lucide.createIcons();
      renderPresetButtons();
      renderBenchmarkFields();
      updateRoiCalculation();
      loadPreset(0);

      // Listen for engine select change
      document.getElementById("engine-select").addEventListener("change", (e) => {
        const keyGroup = document.getElementById("openrouter-key-group");
        if (e.target.value === "openrouter") {
          keyGroup.classList.remove("hidden");
        } else {
          keyGroup.classList.add("hidden");
        }
      });
    });

    // Render presets
    function renderPresetButtons() {
      const container = document.getElementById("preset-buttons-container");
      container.innerHTML = "";
      SAMPLES.forEach((s, idx) => {
        const btn = document.createElement("button");
        btn.className = "p-2 rounded-xl bg-[#161f30] hover:bg-gray-800 border border-gray-800 hover:border-purple-500/50 text-left transition group";
        btn.onclick = () => loadPreset(idx);
        btn.innerHTML = `
          <div class="text-[11px] font-semibold text-gray-200 group-hover:text-purple-300 truncate">${s.name}</div>
          <div class="text-[10px] text-gray-500 flex items-center justify-between mt-0.5">
            <span>${s.difficulty}</span>
            <i data-lucide="arrow-right" class="w-3 h-3 text-gray-600 group-hover:text-purple-400"></i>
          </div>
        `;
        container.appendChild(btn);
      });
      lucide.createIcons();
    }

    function loadPreset(idx) {
      const s = SAMPLES[idx];
      if (!s) return;
      document.getElementById("invoice-input").value = s.text;
    }

    function clearInput() {
      document.getElementById("invoice-input").value = "";
    }

    function handleFileUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        document.getElementById("invoice-input").value = event.target.result;
      };
      reader.readAsText(file);
    }

    // Switch Top Tabs
    function switchTab(tabId) {
      const tabs = ["playground", "benchmarks", "calculator", "pipeline", "api"];
      tabs.forEach(t => {
        const sec = document.getElementById("tab-" + t);
        const nav = document.getElementById("nav-" + t);
        if (t === tabId) {
          sec.classList.remove("hidden");
          if (nav) {
            nav.className = "tab-btn px-4 py-1.5 text-xs font-medium rounded-lg text-white bg-purple-600 shadow transition-all";
          }
        } else {
          sec.classList.add("hidden");
          if (nav) {
            nav.className = "tab-btn px-4 py-1.5 text-xs font-medium rounded-lg text-gray-400 hover:text-white transition-all";
          }
        }
      });
      lucide.createIcons();
    }

    // Switch Result View (Card vs JSON vs Schema)
    function setResultView(view) {
      const cardView = document.getElementById("result-card-view");
      const jsonView = document.getElementById("result-json-view");
      const schemaView = document.getElementById("result-schema-view");

      const btnCard = document.getElementById("btn-view-card");
      const btnJson = document.getElementById("btn-view-json");
      const btnSchema = document.getElementById("btn-view-schema");

      [cardView, jsonView, schemaView].forEach(el => el.classList.add("hidden"));
      [btnCard, btnJson, btnSchema].forEach(b => b.className = "px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 text-gray-300 hover:text-white flex items-center gap-1.5 transition");

      if (view === "card") {
        cardView.classList.remove("hidden");
        btnCard.className = "px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-600 text-white flex items-center gap-1.5 transition";
      } else if (view === "json") {
        jsonView.classList.remove("hidden");
        btnJson.className = "px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-600 text-white flex items-center gap-1.5 transition";
      } else if (view === "schema") {
        schemaView.classList.remove("hidden");
        btnSchema.className = "px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-600 text-white flex items-center gap-1.5 transition";
      }
    }

    // Main Extraction Runner
    async function runExtraction() {
      const text = document.getElementById("invoice-input").value.trim();
      if (!text) {
        alert("Please enter or select an invoice document first.");
        return;
      }

      const engine = document.getElementById("engine-select").value;
      const apiKey = document.getElementById("openrouter-key-input").value.trim();

      const btn = document.getElementById("extract-btn");
      const btnText = document.getElementById("extract-btn-text");
      const emptyState = document.getElementById("result-empty-state");
      const loadingState = document.getElementById("result-loading-state");
      const cardView = document.getElementById("result-card-view");
      const jsonView = document.getElementById("result-json-view");
      const schemaView = document.getElementById("result-schema-view");
      const telemetry = document.getElementById("telemetry-bar");

      btn.disabled = true;
      btnText.innerText = "Extracting...";
      emptyState.classList.add("hidden");
      cardView.classList.add("hidden");
      jsonView.classList.add("hidden");
      schemaView.classList.add("hidden");
      loadingState.classList.remove("hidden");

      try {
        const resp = await fetch("/extract", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            document: text,
            engine: engine,
            api_key: apiKey || undefined
          })
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ detail: "Unknown server error" }));
          throw new Error(err.detail || `Server returned ${resp.status}`);
        }

        const data = await resp.json();
        currentExtraction = data.extraction;

        // Render card
        renderExtractionCard(data.extraction);

        // Render JSON
        document.getElementById("json-code-block").innerText = JSON.stringify(data.extraction, null, 2);

        // Render Telemetry
        telemetry.classList.remove("hidden");
        document.getElementById("telemetry-engine").innerText = data.model;
        document.getElementById("telemetry-latency").innerText = `${data.latency_ms} ms`;

        loadingState.classList.add("hidden");
        setResultView("card");
      } catch (err) {
        loadingState.classList.add("hidden");
        emptyState.classList.remove("hidden");
        alert("Extraction Error: " + err.message);
      } finally {
        btn.disabled = false;
        btnText.innerText = "Extract Structured Data";
        lucide.createIcons();
      }
    }

    // Render extraction into Visual Invoice Card
    function renderExtractionCard(ext) {
      document.getElementById("card-vendor-name").innerText = ext.vendor_name || "Unknown Vendor";
      document.getElementById("card-vendor-address").innerText = ext.vendor_address || "Address not specified";
      document.getElementById("card-invoice-number").innerText = ext.invoice_number ? `#${ext.invoice_number}` : "N/A";
      document.getElementById("card-invoice-date").innerText = ext.invoice_date || "—";
      document.getElementById("card-due-date").innerText = ext.due_date || "—";
      document.getElementById("card-payment-terms").innerText = ext.payment_terms || "Standard";
      document.getElementById("card-currency-badge").innerText = ext.currency || "USD";

      // Bill To
      const billToSec = document.getElementById("card-bill-to-section");
      if (ext.bill_to) {
        billToSec.classList.remove("hidden");
        document.getElementById("card-bill-to").innerText = ext.bill_to;
      } else {
        billToSec.classList.add("hidden");
      }

      // Line Items
      const tbody = document.getElementById("card-line-items-tbody");
      tbody.innerHTML = "";
      const items = ext.line_items || [];
      document.getElementById("card-items-count").innerText = `${items.length} items`;

      items.forEach(item => {
        const tr = document.createElement("tr");
        tr.className = "hover:bg-gray-800/30 transition";
        const sym = getCurrencySymbol(ext.currency);
        tr.innerHTML = `
          <td class="py-2.5 px-4 text-gray-200">${escapeHtml(item.description || "Item")}</td>
          <td class="py-2.5 px-3 text-right text-gray-400">${item.quantity != null ? item.quantity : "—"}</td>
          <td class="py-2.5 px-3 text-right text-gray-400">${item.unit_price != null ? `${sym}${item.unit_price.toFixed(2)}` : "—"}</td>
          <td class="py-2.5 px-4 text-right text-purple-300 font-semibold">${sym}${(item.total || 0).toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
      });

      // Totals
      const sym = getCurrencySymbol(ext.currency);
      document.getElementById("card-subtotal").innerText = ext.subtotal != null ? `${sym}${ext.subtotal.toFixed(2)}` : "—";

      const taxLabel = document.getElementById("card-tax-label");
      if (ext.tax_rate != null) {
        taxLabel.innerHTML = `Tax <span class="text-[10px] px-1.5 py-0.2 bg-gray-800 text-purple-300 rounded font-mono">${(ext.tax_rate * 100).toFixed(1)}%</span>:`;
      } else {
        taxLabel.innerText = "Tax:";
      }
      document.getElementById("card-tax-amount").innerText = ext.tax_amount != null ? `${sym}${ext.tax_amount.toFixed(2)}` : "—";
      document.getElementById("card-grand-total").innerText = `${sym}${(ext.total_amount || 0).toFixed(2)} ${ext.currency || ""}`;

      // Notes
      const notesRow = document.getElementById("card-notes-row");
      if (ext.notes) {
        notesRow.classList.remove("hidden");
        document.getElementById("card-notes").innerText = ext.notes;
      } else {
        notesRow.classList.add("hidden");
      }
    }

    function getCurrencySymbol(curr) {
      if (!curr) return "$";
      const c = curr.toUpperCase();
      if (c === "EUR") return "€";
      if (c === "GBP") return "£";
      if (c === "CAD") return "C$";
      if (c === "AUD") return "A$";
      return "$";
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function copyJsonOutput() {
      if (!currentExtraction) return;
      navigator.clipboard.writeText(JSON.stringify(currentExtraction, null, 2));
      alert("JSON copied to clipboard!");
    }

    function downloadJsonOutput() {
      if (!currentExtraction) return;
      const blob = new Blob([JSON.stringify(currentExtraction, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `invoice_extraction_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }

    function copySnippet(elementId) {
      const text = document.getElementById(elementId).innerText;
      navigator.clipboard.writeText(text);
      alert("Code snippet copied!");
    }

    // Render Benchmark Bars
    function renderBenchmarkFields() {
      const container = document.getElementById("benchmark-fields-container");
      container.innerHTML = "";
      BENCHMARK_FIELDS.forEach(f => {
        const row = document.createElement("div");
        row.className = "p-2 rounded-lg bg-[#0b0f19] border border-gray-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2";
        row.innerHTML = `
          <div class="w-40 font-mono ${f.highlight ? 'text-purple-300 font-bold' : 'text-gray-300'}">
            ${f.name}
          </div>
          <div class="flex-1 flex items-center gap-3">
            <div class="w-14 text-right text-gray-500 font-mono">${f.base}%</div>
            <div class="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden flex">
              <div class="bg-gray-600 h-full" style="width: ${f.base}%"></div>
              <div class="bg-gradient-to-r from-purple-500 to-indigo-500 h-full" style="width: ${f.ft - f.base}%"></div>
            </div>
            <div class="w-14 text-purple-300 font-mono font-bold">${f.ft}%</div>
          </div>
          <div class="w-20 text-right">
            <span class="px-2 py-0.5 rounded text-[11px] font-mono font-semibold ${f.highlight ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/80' : 'bg-gray-800 text-gray-400'}">
              ${f.delta}
            </span>
          </div>
        `;
        container.appendChild(row);
      });
    }

    // ROI Calculator
    function updateRoiCalculation() {
      const slider = document.getElementById("calc-volume-slider");
      const vol = parseInt(slider.value, 10);
      document.getElementById("calc-volume-display").innerText = `${vol.toLocaleString()} invoices/mo`;

      const frontierCostPer1k = 0.35; // OpenRouter / DeepSeek / Claude avg per 1k docs
      const buildCost = 0.40; // Total pipeline build cost

      const monthlyFrontierCost = (vol / 1000) * frontierCostPer1k;
      const monthlySelfHostedCost = 0.00; // Zero marginal cost
      const monthlySavings = monthlyFrontierCost - monthlySelfHostedCost;
      const annualSavings = monthlySavings * 12;

      const breakevenDocs = (buildCost / frontierCostPer1k) * 1000;
      const dailyVol = vol / 30.0;
      const paybackDays = breakevenDocs / dailyVol;

      document.getElementById("calc-monthly-savings").innerText = `$${monthlySavings.toFixed(2)}`;
      document.getElementById("calc-annual-savings").innerText = `$${annualSavings.toFixed(2)}`;
      document.getElementById("calc-breakeven-docs").innerText = Math.ceil(breakevenDocs).toLocaleString();

      if (paybackDays < 1) {
        document.getElementById("calc-payback-days").innerText = `${Math.ceil(paybackDays * 24)} Hours`;
      } else {
        document.getElementById("calc-payback-days").innerText = `~${paybackDays.toFixed(1)} Days`;
      }
    }
  </script>
</body>
</html>
"""
