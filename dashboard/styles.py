"""
Dashboard Styling and Visual Language.
Implements a Premium Dark Navy / Charcoal Emergency Command Center Theme
for the IEEE-Paper AI Evacuation System Demonstration.
"""

def get_custom_css() -> str:
    return """
    <style>
        /* =====================================================================
           1. ROOT & GLOBAL THEME (Dark Navy / Charcoal Command Center)
           ===================================================================== */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-base: #070B14;
            --bg-surface: #0D1424;
            --bg-card: #121C30;
            --bg-card-hover: #17243E;
            --border-subtle: #1C2B47;
            --border-highlight: #283D63;
            --cyan-glow: #00F0FF;
            --cyan-dim: #0284C7;
            --emerald-safe: #10B981;
            --amber-warn: #F59E0B;
            --crimson-hazard: #EF4444;
            --purple-route: #8B5CF6;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
        }

        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: var(--text-primary);
        }

        .stApp {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.04) 0%, transparent 40%);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0A0F1D !important;
            border-right: 1px solid var(--border-subtle) !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Main Container Width & Spacing */
        .main .block-container {
            max-width: 98% !important;
            padding-top: 1rem !important;
            padding-bottom: 3rem !important;
        }

        /* Section Headings */
        .section-header {
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            color: #38BDF8;
            border-left: 3px solid #00F0FF;
            padding-left: 10px;
            margin: 22px 0 10px 0;
            display: flex;
            align-items: center;
        }

        /* =====================================================================
           2. COMMAND CENTER HEADER BANNER
           ===================================================================== */
        .command-header {
            background: linear-gradient(135deg, #0D182B 0%, #08101E 100%);
            border: 1px solid var(--border-highlight);
            border-radius: 10px;
            padding: 18px 24px;
            margin-bottom: 14px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
            position: relative;
            overflow: hidden;
        }
        .command-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, #00F0FF, #3B82F6, #10B981, #00F0FF);
        }
        .command-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #00F0FF;
            background: rgba(0, 240, 255, 0.1);
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-radius: 4px;
            padding: 2px 8px;
            margin-bottom: 6px;
        }
        .command-title {
            font-size: 1.75rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            line-height: 1.25;
            margin: 0 0 4px 0;
        }
        .command-subtitle {
            font-size: 0.88rem;
            font-weight: 600;
            color: #94A3B8;
            letter-spacing: 0.6px;
            margin: 0 0 10px 0;
        }
        .header-meta-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.07);
        }

        /* Status Pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            padding: 3px 10px;
            border-radius: 4px;
        }
        .pill-green {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .pill-blue {
            background: rgba(14, 165, 233, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(14, 165, 233, 0.4);
        }
        .pill-purple {
            background: rgba(139, 92, 246, 0.15);
            color: #A78BFA;
            border: 1px solid rgba(139, 92, 246, 0.4);
        }
        .pill-amber {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }
        .pill-red {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }

        /* =====================================================================
           3. SYSTEM STATUS BAR (Section 1)
           ===================================================================== */
        .status-bar-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
            margin-bottom: 14px;
        }
        .status-bar-item {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 8px 12px;
            text-align: center;
        }
        .status-bar-lbl {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-bar-val {
            font-size: 0.95rem;
            font-weight: 800;
            margin-top: 2px;
            letter-spacing: 0.3px;
        }

        /* =====================================================================
           4. METRIC CARDS & GLASS PANELS
           ===================================================================== */
        .cmd-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
            transition: border-color 0.2s ease;
        }
        .cmd-card:hover {
            border-color: var(--border-highlight);
        }

        .metric-card {
            background: linear-gradient(180deg, #131E33 0%, #0E1727 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 12px 14px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.70rem;
            font-weight: 700;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }
        .metric-value {
            font-size: 1.60rem;
            font-weight: 800;
            margin-top: 3px;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-footer {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Hazard Zone Focus Card */
        .hazard-card-emergency {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(18, 28, 48, 0.85) 100%);
            border-left: 4px solid var(--crimson-hazard);
            border-top: 1px solid rgba(239, 68, 68, 0.3);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }
        .hazard-card-normal {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(18, 28, 48, 0.85) 100%);
            border-left: 4px solid var(--emerald-safe);
            border-top: 1px solid rgba(16, 185, 129, 0.3);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }

        /* CCTV Meta Bar */
        .cctv-meta-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #0A101C;
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 6px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 8px;
        }
        .cctv-meta-bar strong {
            color: #E2E8F0;
        }

        /* =====================================================================
           5. SIMULATED DIGITAL LED SIGNAGE (Section 14)
           ===================================================================== */
        .led-board {
            background: #050810;
            border: 2px solid #1E293B;
            border-radius: 8px;
            padding: 12px 14px;
            text-align: center;
            box-shadow: inset 0 0 14px rgba(0, 0, 0, 0.9);
            font-family: 'JetBrains Mono', monospace;
        }
        .led-title {
            font-size: 0.70rem;
            font-weight: 700;
            color: #64748B;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .led-sign-safe {
            font-size: 1.15rem;
            font-weight: 800;
            color: #00FF88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.7);
            letter-spacing: 1.5px;
            padding: 6px;
            background: rgba(0, 255, 136, 0.06);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 4px;
        }
        .led-sign-avoid {
            font-size: 1.15rem;
            font-weight: 800;
            color: #FF2E4D;
            text-shadow: 0 0 10px rgba(255, 46, 77, 0.7);
            letter-spacing: 1.5px;
            padding: 6px;
            background: rgba(255, 46, 77, 0.06);
            border: 1px solid rgba(255, 46, 77, 0.3);
            border-radius: 4px;
        }
        .led-sign-standby {
            font-size: 1.15rem;
            font-weight: 800;
            color: #94A3B8;
            letter-spacing: 1.5px;
            padding: 6px;
            background: rgba(148, 163, 184, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 4px;
        }

        /* PA Advisory Box (Section 15) */
        .pa-box {
            background: linear-gradient(135deg, #131E33 0%, #0D1626 100%);
            border-left: 4px solid #38BDF8;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 18px;
            font-family: 'Inter', sans-serif;
            font-size: 0.92rem;
            color: #E2E8F0;
            line-height: 1.5;
        }

        /* =====================================================================
           6. OCCUPANCY & PROGRESS BARS (Section 6)
           ===================================================================== */
        .occ-bar-bg {
            background: #1E293B;
            border-radius: 4px;
            height: 8px;
            width: 100%;
            overflow: hidden;
            margin-top: 4px;
        }
        .occ-bar-fill-green {
            background: linear-gradient(90deg, #10B981, #34D399);
            height: 100%;
        }
        .occ-bar-fill-amber {
            background: linear-gradient(90deg, #F59E0B, #FBBF24);
            height: 100%;
        }
        .occ-bar-fill-red {
            background: linear-gradient(90deg, #EF4444, #F87171);
            height: 100%;
        }

        /* =====================================================================
           7. ROUTE INTELLIGENCE & FLOW (Section 9 & 10)
           ===================================================================== */
        .route-path-container {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            padding: 10px 14px;
            background: #0A111F;
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            font-weight: 700;
        }
        .route-node-pill {
            background: #17253D;
            color: #E2E8F0;
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid #283D63;
        }
        .route-node-origin {
            background: rgba(245, 158, 11, 0.2);
            color: #FCD34D;
            border: 1px solid #F59E0B;
        }
        .route-node-exit {
            background: rgba(16, 185, 129, 0.2);
            color: #6EE7B7;
            border: 1px solid #10B981;
        }
        .route-arrow {
            color: #38BDF8;
            font-weight: 800;
            font-size: 1.1rem;
        }

        /* =====================================================================
           8. INCIDENT TIMELINE (Unique Feature 3)
           ===================================================================== */
        .timeline-box {
            background: #0B1220;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 18px;
        }
        .timeline-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding-bottom: 10px;
            position: relative;
        }
        .timeline-item:not(:last-child)::after {
            content: '';
            position: absolute;
            left: 18px; top: 22px; bottom: 0;
            width: 2px;
            background: var(--border-highlight);
        }
        .timeline-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #00F0FF;
            margin-top: 5px;
            flex-shrink: 0;
            box-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
        }
        .timeline-time {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #94A3B8;
            min-width: 65px;
            font-weight: 600;
        }
        .timeline-desc {
            font-size: 0.85rem;
            color: #F1F5F9;
        }

        /* =====================================================================
           9. STREAMLIT WIDGET OVERRIDES FOR PREMIUM APPEARANCE
           ===================================================================== */
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(180deg, #18263E 0%, #101B2E 100%) !important;
            color: #F8FAFC !important;
            border: 1px solid #2A3E62 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            letter-spacing: 0.4px !important;
            padding: 6px 14px !important;
            transition: all 0.15s ease !important;
        }
        div.stButton > button:hover {
            border-color: #00F0FF !important;
            color: #00F0FF !important;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.25) !important;
        }
        div.stButton > button:active {
            transform: scale(0.98);
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-subtle) !important;
            border-radius: 6px !important;
            overflow: hidden;
        }

        /* Progress Bar */
        div.stProgress > div > div > div > div {
            background-color: #00F0FF !important;
        }

        /* High-Contrast Tabs Styling */
        div[data-baseweb="tab-list"] {
            gap: 6px !important;
            background-color: #0A101D !important;
            padding: 6px 8px !important;
            border-radius: 8px !important;
            border: 1px solid #1C2D4A !important;
            margin-bottom: 12px !important;
        }
        button[data-baseweb="tab"] {
            color: #CBD5E1 !important;
            background: #0E1729 !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            border-radius: 6px !important;
            border: 1px solid #1E2D48 !important;
            padding: 8px 14px !important;
            transition: all 0.2s ease !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #00F0FF !important;
            background: rgba(0, 240, 255, 0.12) !important;
            border-color: #00F0FF !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00F0FF !important;
            background: rgba(0, 240, 255, 0.18) !important;
            border-color: #00F0FF !important;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.2) !important;
        }

        /* Widget Labels & Text High-Contrast */
        label, .stSelectbox label, .stSlider label, .stRadio label, .stTextInput label, .stFileUploader label {
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        /* =====================================================================
           DROPDOWN & SELECTBOX THEME OVERRIDES (React-Aria & BaseWeb)
           Eliminates white-on-white text; enforces dark command-center aesthetic
           ===================================================================== */
        /* 1. Closed Selectbox / Combobox Input Box */
        div[data-testid="stSelectbox"] div[role="group"],
        div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"],
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[role="group"] {
            background-color: #0E1729 !important;
            border: 1px solid #1C2D4A !important;
            border-radius: 6px !important;
            color: #F8FAFC !important;
        }

        /* Hover & Focus state on Selectbox Control */
        div[data-testid="stSelectbox"] div[role="group"]:hover,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:hover {
            border-color: #283D63 !important;
        }
        div[data-testid="stSelectbox"] div[role="group"]:focus-within,
        div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"]:focus-within > div {
            border-color: #00F0FF !important;
            box-shadow: 0 0 0 1px #00F0FF, 0 0 8px rgba(0, 240, 255, 0.25) !important;
        }

        /* 2. Text Inside Closed Selectbox / Input */
        div[data-testid="stSelectbox"] input,
        div[data-testid="stSelectbox"] input[role="combobox"],
        div[data-testid="stMultiSelect"] input,
        div[data-testid="stSelectbox"] div[data-testid="stMarkdownContainer"],
        div[data-testid="stSelectbox"] span {
            color: #F8FAFC !important;
            background-color: transparent !important;
            font-size: 0.85rem !important;
        }
        div[data-testid="stSelectbox"] input::placeholder,
        div[data-testid="stMultiSelect"] input::placeholder {
            color: #64748B !important;
        }

        /* Dropdown Chevron / Arrow Icon & Buttons */
        div[data-testid="stSelectbox"] button,
        div[data-testid="stSelectbox"] button[aria-label="Open"],
        div[data-testid="stSelectbox"] div[role="group"] > button {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        div[data-testid="stSelectbox"] button svg,
        div[data-testid="stSelectbox"] svg,
        div[data-testid="stMultiSelect"] svg {
            fill: #94A3B8 !important;
            color: #94A3B8 !important;
        }
        div[data-testid="stSelectbox"]:hover svg,
        div[data-testid="stMultiSelect"]:hover svg {
            fill: #00F0FF !important;
            color: #00F0FF !important;
        }

        /* 3. Popover Portal Container (Dropdown Menu) */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div:has(> div[role="listbox"]),
        div:has(> ul[role="listbox"]),
        .react-aria-Popover,
        [data-testid="stSelectboxVirtualDropdown"] {
            background-color: #111827 !important;
            border: 1px solid #1E2D48 !important;
            border-radius: 6px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7), 0 0 10px rgba(0, 240, 255, 0.1) !important;
            overflow: hidden !important;
        }

        /* 4. Listbox Container */
        div[role="listbox"],
        ul[role="listbox"],
        ul[data-baseweb="menu"] {
            background-color: #111827 !important;
            padding: 4px !important;
            border: none !important;
        }

        /* 5. Dropdown Option Items (Normal / Unselected) */
        div[role="option"],
        li[role="option"],
        li[data-baseweb="menu-item"] {
            background-color: transparent !important;
            color: #F8FAFC !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            padding: 8px 12px !important;
            border-radius: 4px !important;
            margin: 2px 0 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
        }
        div[role="option"] > div,
        li[role="option"] > div,
        div[role="option"] span,
        li[role="option"] span {
            color: #F8FAFC !important;
        }

        /* 6. Option Hover & Focused State */
        div[role="option"]:hover,
        li[role="option"]:hover,
        li[data-baseweb="menu-item"]:hover,
        div[role="option"][data-focused="true"],
        li[role="option"][data-focused="true"],
        div[role="option"]:focus,
        li[role="option"]:focus {
            background-color: #1E293B !important;
            color: #38BDF8 !important;
        }
        div[role="option"]:hover *,
        li[role="option"]:hover * {
            color: #38BDF8 !important;
        }

        /* 7. Selected Option State */
        div[role="option"][aria-selected="true"],
        li[role="option"][aria-selected="true"],
        div[role="option"][data-selected="true"],
        li[role="option"][data-selected="true"] {
            background-color: rgba(0, 240, 255, 0.15) !important;
            color: #00F0FF !important;
            font-weight: 700 !important;
            border-left: 3px solid #00F0FF !important;
        }
        div[role="option"][aria-selected="true"] *,
        li[role="option"][aria-selected="true"] * {
            color: #00F0FF !important;
            font-weight: 700 !important;
        }
        div[role="option"][aria-selected="true"]:hover,
        li[role="option"][aria-selected="true"]:hover {
            background-color: rgba(0, 240, 255, 0.25) !important;
            color: #00F0FF !important;
        }

        /* 8. Multiselect Tag Badges */
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background-color: #17253D !important;
            color: #00F0FF !important;
            border: 1px solid #283D63 !important;
            border-radius: 4px !important;
        }
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] * {
            color: #00F0FF !important;
        }
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {
            fill: #94A3B8 !important;
        }
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg:hover {
            fill: #EF4444 !important;
        }

        /* Expander */
        details {
            background: #0A111F !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 6px !important;
            margin-bottom: 8px !important;
        }
        summary {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
        }

        /* Disclaimer & Model Accuracy Banners */
        .disclaimer-banner {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.4);
            border-radius: 6px;
            padding: 8px 16px;
            color: #FCA5A5;
            font-size: 0.80rem;
            font-weight: 700;
            text-align: center;
            letter-spacing: 0.8px;
            margin-bottom: 8px;
        }
        .model-accuracy-banner {
            background: rgba(14, 165, 233, 0.1);
            border: 1px solid rgba(14, 165, 233, 0.3);
            border-radius: 6px;
            padding: 6px 14px;
            color: #38BDF8;
            font-size: 0.78rem;
            font-weight: 700;
            text-align: center;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        /* Evacuation Risk Index Cards (Unique Feature 1) */
        .risk-card-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, #121C30 100%);
            border-left: 4px solid #10B981;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 18px;
        }
        .risk-card-moderate {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, #121C30 100%);
            border-left: 4px solid #F59E0B;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 18px;
        }
        .risk-card-high {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.18) 0%, #121C30 100%);
            border-left: 4px solid #F97316;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 18px;
        }
        .risk-card-critical {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, #121C30 100%);
            border-left: 4px solid #EF4444;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px 18px;
        }

        /* 3-Layer Architecture Diagram (Section 21) */
        .arch-container {
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            gap: 10px;
            margin: 14px 0;
            flex-wrap: wrap;
        }
        .arch-layer {
            flex: 1;
            min-width: 200px;
            background: #0E1729;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 12px 14px;
        }
        .arch-layer-title {
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border-highlight);
        }
        .arch-item {
            background: #152238;
            border: 1px solid #1F3354;
            border-radius: 4px;
            padding: 6px 10px;
            margin-bottom: 6px;
            font-size: 0.80rem;
            color: #E2E8F0;
        }
        .arch-arrow-block {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #00F0FF;
            font-size: 1.4rem;
            font-weight: 800;
        }

        /* Formula & Math Box (Section 22) */
        .math-box {
            background: #090F1C;
            border: 1px solid #1C2D4A;
            border-radius: 8px;
            padding: 14px 18px;
            margin: 10px 0;
        }
    </style>
    """
