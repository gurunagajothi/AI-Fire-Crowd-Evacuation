"""
Dashboard Styling and Visual Language.
Implements Clean, Elegant, Modern Dual-Theme (Light & Dark) Emergency-Management Styling
for the AI Evacuation Assistant & IEEE Demonstration.
"""

def get_custom_css(dark_mode: bool = False) -> str:
    """
    Returns custom CSS styles for the application.
    Supports both Light Mode (default) and Dark Mode (command-center theme)
    with full contrast, readable typography, and high-visibility status styling.
    """
    if dark_mode:
        # Dark Command-Center Theme Variables
        theme_vars = """
            --bg-base: #0B0F19;
            --bg-surface: #111827;
            --bg-card: #151E32;
            --bg-card-hover: #1E293B;
            --border-subtle: #243048;
            --border-strong: #334155;
            --primary-blue: #3B82F6;
            --primary-blue-hover: #60A5FA;
            --blue-soft: rgba(59, 130, 246, 0.15);
            --emerald-safe: #10B981;
            --emerald-soft: rgba(16, 185, 129, 0.15);
            --emerald-border: #059669;
            --amber-warn: #F59E0B;
            --amber-soft: rgba(245, 158, 11, 0.15);
            --amber-border: #D97706;
            --crimson-hazard: #EF4444;
            --crimson-soft: rgba(239, 68, 68, 0.15);
            --crimson-border: #DC2626;
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #94A3B8;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
            --card-shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
            --sidebar-bg: #111827;
            --popover-bg: #151E32;
            --popover-text: #F8FAFC;
            --popover-hover: #1E293B;
            --code-bg: #1E293B;
            --code-text: #38BDF8;
        """
        sidebar_bg = "#111827"
        header_bg = "#151E32"
        card_bg = "#151E32"
        breadcrumbs_bg = "#111827"
        pill_bg = "#1E293B"
        pill_text = "#E2E8F0"
        pill_border = "#334155"
        select_bg = "#151E32"
        select_text = "#F8FAFC"
        select_border = "#334155"
    else:
        # Modern Light Theme Variables
        theme_vars = """
            --bg-base: #F8FAFC;
            --bg-surface: #FFFFFF;
            --bg-card: #FFFFFF;
            --bg-card-hover: #F1F5F9;
            --border-subtle: #E2E8F0;
            --border-strong: #CBD5E1;
            --primary-blue: #2563EB;
            --primary-blue-hover: #1D4ED8;
            --blue-soft: #EFF6FF;
            --emerald-safe: #16A34A;
            --emerald-soft: #F0FDF4;
            --emerald-border: #BBF7D0;
            --amber-warn: #D97706;
            --amber-soft: #FFFBEB;
            --amber-border: #FDE68A;
            --crimson-hazard: #DC2626;
            --crimson-soft: #FEF2F2;
            --crimson-border: #FECACA;
            --text-primary: #0F172A;
            --text-secondary: #475569;
            --text-muted: #64748B;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            --card-shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
            --sidebar-bg: #FFFFFF;
            --popover-bg: #FFFFFF;
            --popover-text: #0F172A;
            --popover-hover: #F1F5F9;
            --code-bg: #F1F5F9;
            --code-text: #1E40AF;
        """
        sidebar_bg = "#FFFFFF"
        header_bg = "#FFFFFF"
        card_bg = "#FFFFFF"
        breadcrumbs_bg = "#FFFFFF"
        pill_bg = "#F1F5F9"
        pill_text = "#334155"
        pill_border = "#CBD5E1"
        select_bg = "#FFFFFF"
        select_text = "#0F172A"
        select_border = "#CBD5E1"

    return f"""
    <style>
        /* =====================================================================
           1. ROOT & GLOBAL THEME
           ===================================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

        :root {{
            {theme_vars}
        }}

        html, body, .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }}

        /* Material Symbols ligature preservation */
        [data-testid="stIconMaterial"],
        span[data-testid="stIconMaterial"],
        .material-symbols-rounded {{
            font-family: "Material Symbols Rounded" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            display: inline-block !important;
            white-space: nowrap !important;
            direction: ltr !important;
            font-feature-settings: "liga" !important;
            -webkit-font-feature-settings: "liga" !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            border-right: 1px solid var(--border-subtle) !important;
            box-shadow: 1px 0 3px rgba(0,0,0,0.02) !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {{
            color: var(--text-primary) !important;
        }}

        /* Main Container Width & Spacing */
        .main .block-container {{
            max-width: 1200px !important;
            padding-top: 1.2rem !important;
            padding-bottom: 3.5rem !important;
        }}

        /* =====================================================================
           2. POLISHED HEADER & STATUS BADGES
           ===================================================================== */
        .app-header {{
            background: {header_bg};
            border: 1px solid var(--border-subtle);
            border-top: 3px solid var(--primary-blue);
            border-radius: 12px;
            padding: 16px 24px;
            margin-bottom: 20px;
            box-shadow: var(--card-shadow);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .app-title {{
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.4px;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .app-subtitle {{
            font-size: 0.88rem;
            font-weight: 500;
            color: var(--text-muted);
            margin: 2px 0 0 0;
        }}

        /* Status Pills */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.80rem;
            font-weight: 700;
            padding: 6px 12px;
            border-radius: 20px;
            letter-spacing: 0.2px;
        }}
        .status-ready {{
            background: var(--emerald-soft);
            color: var(--emerald-safe);
            border: 1px solid var(--emerald-border);
        }}
        .status-evaluating {{
            background: var(--blue-soft);
            color: var(--primary-blue);
            border: 1px solid var(--border-subtle);
        }}
        .status-emergency {{
            background: var(--crimson-soft);
            color: var(--crimson-hazard);
            border: 1px solid var(--crimson-border);
            animation: pulse-soft 2s infinite;
        }}

        @keyframes pulse-soft {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.82; }}
        }}

        /* =====================================================================
           3. HOME CHOICE CARDS (Side-by-Side Analysis Selector)
           ===================================================================== */
        .home-choice-card {{
            background: {card_bg};
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 24px 22px;
            text-align: center;
            box-shadow: var(--card-shadow);
            transition: all 0.2s ease-in-out;
            margin-bottom: 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 220px;
        }}
        .home-choice-card:hover {{
            border-color: var(--primary-blue);
            box-shadow: var(--card-shadow-hover);
            transform: translateY(-2px);
        }}
        .home-card-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 6px;
        }}
        .home-card-desc {{
            font-size: 0.90rem;
            color: var(--text-muted);
            line-height: 1.45;
            margin-bottom: 16px;
        }}

        /* =====================================================================
           4. PROMINENT EVACUATION GUIDANCE BANNER (Primary Result Feature)
           ===================================================================== */
        .evac-guidance-emergency {{
            background: var(--crimson-soft);
            border-left: 6px solid var(--crimson-hazard);
            border-top: 1px solid var(--crimson-border);
            border-right: 1px solid var(--crimson-border);
            border-bottom: 1px solid var(--crimson-border);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.1);
        }}
        .evac-guidance-normal {{
            background: var(--emerald-soft);
            border-left: 6px solid var(--emerald-safe);
            border-top: 1px solid var(--emerald-border);
            border-right: 1px solid var(--emerald-border);
            border-bottom: 1px solid var(--emerald-border);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(22, 163, 74, 0.1);
        }}
        .evac-guidance-caution {{
            background: var(--amber-soft);
            border-left: 6px solid var(--amber-warn);
            border-top: 1px solid var(--amber-border);
            border-right: 1px solid var(--amber-border);
            border-bottom: 1px solid var(--amber-border);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(217, 119, 6, 0.1);
        }}
        .evac-header-tag {{
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .evac-exit-title {{
            font-size: 1.85rem;
            font-weight: 800;
            margin: 4px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-primary);
        }}

        /* Route Pathway Flow Breadcrumbs */
        .route-flow-container {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin: 12px 0;
            padding: 12px 14px;
            background: {breadcrumbs_bg};
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
        }}
        .route-node-pill {{
            background: {pill_bg};
            color: {pill_text};
            border: 1px solid {pill_border};
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .route-node-origin {{
            background: var(--blue-soft);
            color: var(--primary-blue);
            border: 1px solid var(--border-subtle);
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .route-node-exit {{
            background: var(--emerald-soft);
            color: var(--emerald-safe);
            border: 1px solid var(--emerald-border);
            padding: 5px 14px;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }}
        .route-arrow {{
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 700;
        }}
        .evac-action-instruction {{
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-top: 8px;
            line-height: 1.45;
        }}

        /* =====================================================================
           5. COMPACT METRIC CARDS & SUMMARY
           ===================================================================== */
        .clean-metric-card {{
            background: {card_bg};
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: var(--card-shadow);
            text-align: left;
            margin-bottom: 10px;
        }}
        .clean-metric-label {{
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .clean-metric-value {{
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            align-items: baseline;
            gap: 8px;
        }}
        .clean-metric-tag {{
            font-size: 0.72rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 12px;
        }}

        /* Legacy Compatibility Classes */
        .cmd-card,
        .metric-card,
        .signage-card {{
            background: {card_bg};
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: var(--card-shadow);
            margin-bottom: 12px;
        }}
        .metric-label {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-top: 3px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .metric-footer {{
            font-size: 0.74rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .section-header {{
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            color: var(--text-primary);
            border-left: 4px solid var(--primary-blue);
            padding-left: 10px;
            margin: 20px 0 12px 0;
            display: flex;
            align-items: center;
        }}

        .pa-message-box {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.90rem;
            color: var(--text-primary);
            line-height: 1.45;
        }}

        /* Academic Disclaimer Banner */
        .academic-disclaimer {{
            background: {breadcrumbs_bg};
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 24px;
            text-align: center;
            line-height: 1.4;
        }}

        /* Buttons Styling */
        div.stButton > button {{
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.15s ease !important;
        }}
        div.stButton > button[kind="primary"] {{
            background-color: var(--primary-blue) !important;
            border-color: var(--primary-blue) !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: var(--primary-blue-hover) !important;
            border-color: var(--primary-blue-hover) !important;
            box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3) !important;
        }}

        /* High-contrast Selectbox Popover & Options */
        div[data-baseweb="select"] {{
            background-color: {select_bg} !important;
            border-color: {select_border} !important;
            color: {select_text} !important;
        }}
        div[data-baseweb="popover"],
        ul[role="listbox"],
        li[role="option"] {{
            background-color: {select_bg} !important;
            color: {select_text} !important;
        }}
        li[role="option"]:hover,
        div[role="option"]:hover {{
            background-color: var(--bg-card-hover) !important;
            color: var(--text-primary) !important;
        }}

        /* Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 2px solid var(--border-subtle);
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px 6px 0 0;
            padding: 8px 16px;
            font-weight: 600;
            color: var(--text-muted);
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--primary-blue) !important;
            border-bottom: 2px solid var(--primary-blue) !important;
        }}
    </style>
    """
