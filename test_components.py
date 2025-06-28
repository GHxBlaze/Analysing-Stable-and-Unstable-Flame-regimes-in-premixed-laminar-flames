from timegan_flame_generator import FlameDataProcessor
import os

def test_parameter_extraction():
    print("Testing parameter extraction...")
    
    processor = FlameDataProcessor("")
    
    test_filenames = [
        "Phi_0p8_u_0p6_30s_20250118_154139.txt",
        "Phi_1p2_u_0p7_10s_20250118_161858.txt", 
        "Phi_1p0_u_0p2_10s_20250118_154534.txt"
    ]
    
    for filename in test_filenames:
        try:
            phi, u = processor.extract_parameters_from_filename(filename)
            print(f"  {filename} -> φ={phi}, u={u}")
        except Exception as e:
            print(f"  {filename} -> Error: {e}")
    
    print("✓ Parameter extraction test completed\n")

def test_data_loading():
    print("Testing data loading...")
    
    processor = FlameDataProcessor(".")
    
    txt_files = [f for f in os.listdir(".") if f.endswith('.txt')]
    if txt_files:
        test_file = txt_files[0]
        print(f"  Loading: {test_file}")
        
        data = processor.load_data_file(test_file)
        if data is not None:
            print(f"  ✓ Data shape: {data.shape}")
            print(f"  ✓ Data type: {data.dtype}")
            print(f"  ✓ Sample values:")
            print(f"    Heat: {data[0, 0]:.6E}")
            print(f"    Time: {data[0, 1]:.6E}")
            print(f"    Pressure: {data[0, 2]:.6E}")
        else:
            print("  ✗ Failed to load data")
    else:
        print("  ✗ No .txt files found")
    
    print("✓ Data loading test completed\n")

def test_full_data_loading():
    print("Testing full data loading...")
    
    processor = FlameDataProcessor("./Baseline - Data")
    
    try:
        sequences, parameters = processor.load_all_data()
        print(f"  ✓ Loaded {len(sequences)} sequences")
        print(f"  ✓ Sequences shape: {sequences.shape}")
        print(f"  ✓ Parameters shape: {parameters.shape}")
        print(f"  ✓ Parameter ranges:")
        print(f"    φ: {parameters[:, 0].min():.1f} to {parameters[:, 0].max():.1f}")
        print(f"    u: {parameters[:, 1].min():.1f} to {parameters[:, 1].max():.1f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("✓ Full data loading test completed\n")

if __name__ == "__main__":
    print("=== TimeGAN Component Tests ===\n")
    
    test_parameter_extraction()
    test_data_loading()
    test_full_data_loading()
    
    print("🎉 All tests completed!")
    print("\nTo run the full demo, use: python demo_timegan.py")
    print("To train the model, use: python generate_flame_data.py train")
