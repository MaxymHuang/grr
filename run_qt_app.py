#!/usr/bin/env python3
"""
Simple launcher for the Qt6 Gage R&R Analysis application.
"""

if __name__ == "__main__":
    try:
        from gage_rr_analysis_qt import main
        main()
    except ImportError as e:
        print(f"Error importing Qt6 application: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements.txt")
        print("\nOr with uv:")
        print("uv pip install -r requirements.txt")
    except Exception as e:
        print(f"Error running application: {e}")
