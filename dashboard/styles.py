"""
Dashboard Styling and CSS Definitions.
Provides professional IEEE-academic dark theme styling, digital LED signage themes,
and emergency badge styles.
"""

def get_custom_css() -> str:
    return """
    <style>
        /* Base typography & theme */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #E2E8F0;
        }

        /* Main Header Banner */
        .header-container {
            background: linear-gradient(180deg, #131B26 0%, #0D131D 100%);
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 12px;
        }
        .main-title {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.5px;
            margin: 0 0 6px 0;
            line-height: 1.25;
        }
        .main-subtitle {
            font-size: 0.98rem;
            color: #94A3B8;
            font-weight: 500;
            margin: 0 0 12px 0;
        }
        .system-status-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 8px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .status-pill {
            padding: 4px 10px;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .pill-green {
            background-color: #064E3B;
            color: #34D399;
            border: 1px solid #059669;
        }
        .pill-blue {
            background-color: #1E3A8A;
            color: #60A5FA;
            border: 1px solid #2563EB;
        }
        .pill-amber {
            background-color: #78350F;
            color: #FBBF24;
            border: 1px solid #D97706;
        }

        /* Academic Disclaimer Banner */
        .disclaimer-banner {
            background-color: #0F172A;
            border-left: 4px solid #38BDF8;
            border-radius: 4px;
            padding: 10px 14px;
            margin-bottom: 16px;
            font-size: 0.85rem;
            color: #94A3B8;
            line-height: 1.45;
        }
        .disclaimer-banner strong {
            color: #38BDF8;
        }

        /* Section 2: Compact System Status Cards */
        .sys-status-card {
            background-color: #111827;
            border: 1px solid #1F2937;
            border-radius: 6px;
            padding: 10px 12px;
            text-align: center;
        }
        .sys-status-label {
            font-size: 0.70rem;
            color: #9CA3AF;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .sys-status-value {
            font-size: 1.05rem;
            font-weight: 800;
            margin-top: 2px;
        }

        /* Section Headings */
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #F1F5F9;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 6px;
            margin-top: 24px;
            margin-bottom: 14px;
            letter-spacing: -0.2px;
        }

        /* Section 4: Metadata Bar Below Images */
        .cctv-meta-bar {
            display: flex;
            justify-content: space-around;
            background-color: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 4px;
            padding: 8px 12px;
            margin-top: 8px;
            margin-bottom: 14px;
            font-size: 0.80rem;
            color: #94A3B8;
        }
        .cctv-meta-item strong {
            color: #F8FAFC;
        }

        /* Status Cards */
        .status-card {
            padding: 14px 18px;
            border-radius: 6px;
            color: #FFFFFF;
            font-weight: 600;
            margin-bottom: 14px;
            text-align: center;
        }
        .status-normal {
            background-color: #064E3B;
            border: 1px solid #059669;
        }
        .status-caution {
            background-color: #78350F;
            border: 1px solid #D97706;
        }
        .status-emergency {
            background-color: #7F1D1D;
            border: 1px solid #DC2626;
        }
        .status-rerouted {
            background-color: #7C2D12;
            border: 1px solid #EA580C;
        }

        /* Badges */
        .status-badge {
            padding: 8px 14px;
            border-radius: 6px;
            color: #FFFFFF;
            font-weight: 700;
            text-align: center;
            font-size: 1.05rem;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        .badge-danger {
            background-color: #7F1D1D;
            border: 1px solid #EF4444;
            color: #FCA5A5;
        }
        .badge-warning {
            background-color: #7C2D12;
            border: 1px solid #F97316;
            color: #FDBA74;
        }
        .badge-caution {
            background-color: #78350F;
            border: 1px solid #F59E0B;
            color: #FDE68A;
        }
        .badge-normal {
            background-color: #064E3B;
            border: 1px solid #10B981;
            color: #6EE7B7;
        }
        .badge-rerouted {
            background-color: #4C1D95;
            border: 1px solid #8B5CF6;
            color: #DDD6FE;
        }

        /* Metric Cards */
        .metric-card {
            background-color: #111827;
            border: 1px solid #1F2937;
            border-radius: 6px;
            padding: 12px 14px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #9CA3AF;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #F9FAFB;
            margin-top: 4px;
        }

        /* Route Flow Visualizer */
        .route-flow-container {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            background-color: #0F172A;
            border: 1px solid #1E293B;
            padding: 16px;
            border-radius: 6px;
            margin-top: 10px;
            margin-bottom: 12px;
            gap: 8px;
        }
        .route-node {
            background-color: #1E293B;
            color: #F8FAFC;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.95rem;
            border: 1px solid #334155;
        }
        .route-node-exit {
            background-color: #064E3B;
            border: 1px solid #10B981;
            color: #34D399;
        }
        .route-arrow {
            font-size: 1.25rem;
            color: #38BDF8;
            font-weight: 800;
        }

        /* Simulated Digital LED Signage */
        .led-panel {
            background-color: #020617;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 12px;
        }
        .led-sign {
            font-family: "Courier New", Courier, monospace;
            font-weight: 900;
            font-size: 1.05rem;
            padding: 10px 14px;
            border-radius: 4px;
            text-align: center;
            letter-spacing: 1px;
        }
        .led-green {
            background-color: #022C22;
            color: #34D399;
            border: 1px solid #059669;
        }
        .led-red {
            background-color: #450A0A;
            color: #F87171;
            border: 1px solid #DC2626;
        }
        .led-gray {
            background-color: #111827;
            color: #9CA3AF;
            border: 1px solid #374151;
        }

        /* PA Announcement Box */
        .pa-announcement-box {
            background-color: #1C1917;
            border-left: 4px solid #F59E0B;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.92rem;
            color: #FDE68A;
            font-style: italic;
            line-height: 1.5;
            margin-bottom: 14px;
        }

        /* Comparison Table Box */
        .comparison-box {
            background-color: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 14px;
            margin-bottom: 12px;
        }

        /* Compact Limitations Footer */
        .limitations-footer {
            background-color: #0A0F1D;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 14px 18px;
            margin-top: 25px;
            font-size: 0.84rem;
            color: #94A3B8;
            line-height: 1.6;
        }
        .limitations-footer ol {
            margin: 6px 0 0 16px;
            padding: 0;
        }
    </style>
    """
