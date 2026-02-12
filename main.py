import argparse
import sys
from src.processor import MeasurementProcessor

def main():
    """
    Main entry point for the Automated 3D Measurement Tool.
    This script processes a 3D scan (folder or file) and outputs geometric
    measurements for detected features like trenches or manholes.
    """
    parser = argparse.ArgumentParser(description="Automated 3D Measurement Tool")
    parser.add_argument('--input', '-i', required=True, help='Path to Scaniverse folder or 3D model file')
    parser.add_argument('--out-dir', '-o', default='out', help='Output directory')
    args = parser.parse_args()

    try:
        processor = MeasurementProcessor(args.input, args.out_dir)
        results = processor.process()

        if results.get("status") == "error":
            print(f"Error: {results['message']}")
            sys.exit(1)

        print("\nMeasurement Results:")
        print(f"Detected Type: {results['detected_type']}")
        print(f"Scale Valid: {results['scale_valid']} ({results['validation_reason']})")
        for k, v in results['measurements_m'].items():
            if k == 'confidence':
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v:.4f} m")

    except Exception as e:
        print(f"Error processing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
