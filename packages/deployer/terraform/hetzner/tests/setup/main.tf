# Test fixture: Generate test SSH key for testing
# This creates a temporary SSH key that tests can reference

terraform {
  required_providers {
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}

resource "tls_private_key" "test" {
  algorithm = "ED25519"
}

resource "local_file" "test_public_key" {
  content  = tls_private_key.test.public_key_openssh
  filename = "${path.module}/test_key.pub"
}

resource "local_file" "test_private_key" {
  content         = tls_private_key.test.private_key_openssh
  filename        = "${path.module}/test_key"
  file_permission = "0600"
}

output "test_public_key_path" {
  value = local_file.test_public_key.filename
}

output "test_public_key_content" {
  value = tls_private_key.test.public_key_openssh
}
