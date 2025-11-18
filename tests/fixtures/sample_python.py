"""Sample Python file for testing parser"""



class User:
    """User model class"""

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self) -> str:
        """Return greeting message"""
        return f"Hello, {self.name}!"

    @staticmethod
    def create_default() -> "User":
        return User("Anonymous", 0)


class Admin(User):
    """Admin user class"""

    def __init__(self, name: str, age: int, permissions: list[str]):
        super().__init__(name, age)
        self.permissions = permissions

    async def check_permission(self, permission: str) -> bool:
        """Check if admin has permission"""
        return permission in self.permissions
def calculate_total(items: list[dict]) -> float:
    """Calculate total price from items"""
    total = 0.0
    for item in items:
        total += item.get("price", 0.0)
    return total


async def fetch_data(url: str):
    """Fetch data from URL"""
    # Implementation here
    pass

