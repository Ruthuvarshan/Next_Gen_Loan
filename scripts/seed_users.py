"""
Seed script to populate initial users in the database.
Creates demo users for testing the authentication system.

Usage:
    python scripts/seed_users.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.database import init_databases, UsersSessionLocal, User
from src.utils.auth import get_password_hash


def create_demo_users():
    """Create demo users for testing."""
    print("=" * 60)
    print("SEEDING USER DATABASE")
    print("=" * 60)
    
    # Initialize databases
    print("\n1. Initializing databases...")
    init_databases()
    
    # Create session
    db = UsersSessionLocal()
    
    try:
        # Check if users already exist
        existing_count = db.query(User).count()
        
        if existing_count > 0:
            print(f"\n⚠️  Database already contains {existing_count} users.")
            response = input("Delete existing users and reseed? (yes/no): ")
            
            if response.lower() != "yes":
                print("Aborted.")
                return
            
            # Delete all existing users
            db.query(User).delete()
            db.commit()
            print("✓ Existing users deleted")
        
        # Define demo users
        demo_users = [
            {
                "username": "loan_officer",
                "email": "officer@loansystem.com",
                "password": "officer123",
                "full_name": "Jane Smith",
                "role": "user",
                "is_active": True
            },
            {
                "username": "loan_officer2",
                "email": "officer2@loansystem.com",
                "password": "officer123",
                "full_name": "John Doe",
                "role": "user",
                "is_active": True
            },
            {
                "username": "admin",
                "email": "admin@loansystem.com",
                "password": "admin123",
                "full_name": "System Administrator",
                "role": "admin",
                "is_active": True
            },
            {
                "username": "risk_analyst",
                "email": "analyst@loansystem.com",
                "password": "analyst123",
                "full_name": "Sarah Johnson",
                "role": "admin",
                "is_active": True
            }
        ]
        
        print(f"\n2. Creating {len(demo_users)} demo users...")
        
        for user_data in demo_users:
            # Hash password
            hashed_password = get_password_hash(user_data["password"])
            
            # Create user
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                hashed_password=hashed_password,
                full_name=user_data["full_name"],
                role=user_data["role"],
                is_active=user_data["is_active"]
            )
            
            db.add(user)
            print(f"   ✓ Created {user_data['role']:5s} user: {user_data['username']:15s} (password: {user_data['password']})")
        
        # Commit to database
        db.commit()
        
        print("\n" + "=" * 60)
        print("SEEDING COMPLETE")
        print("=" * 60)
        print("\nYou can now log in with:")
        print("\n📋 LOAN OFFICER ACCOUNTS (User Portal):")
        print("   Username: loan_officer   Password: officer123")
        print("   Username: loan_officer2  Password: officer123")
        print("\n👑 ADMIN ACCOUNTS (Admin Dashboard):")
        print("   Username: admin          Password: admin123")
        print("   Username: risk_analyst   Password: analyst123")
        print("\n⚠️  SECURITY WARNING: Change these passwords in production!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        db.rollback()
        raise
    
    finally:
        db.close()


if __name__ == "__main__":
    create_demo_users()
