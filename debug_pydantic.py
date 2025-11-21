from src.api.schemas import PredictionRequest
from pydantic import ValidationError

def test_schema():
    print("🧪 Testing Pydantic Schema...")
    
    # Test 1: Small list of floats
    try:
        data = {"image": [0.0, 1.0]}
        req = PredictionRequest(**data)
        print("✅ Small list passed")
    except ValidationError as e:
        print(f"❌ Small list failed: {e}")

    # Test 2: List of ints (should cast to float)
    try:
        data = {"image": [0, 1]}
        req = PredictionRequest(**data)
        print("✅ List of ints passed")
    except ValidationError as e:
        print(f"❌ List of ints failed: {e}")

    # Test 3: Large list of zeros
    try:
        data = {"image": [0.0] * 784}
        req = PredictionRequest(**data)
        print("✅ Large list passed")
    except ValidationError as e:
        print(f"❌ Large list failed: {e}")

if __name__ == "__main__":
    test_schema()
