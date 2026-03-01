class DummyInfoModule:
    def get_information(self, context):
        print("[Module 3] Providing external knowledge.")
        return {"instructions": "Sample instructions"}