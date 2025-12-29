# Terraform native tests for GCP module
# Run with: terraform test

# =============================================================================
# Test Variables
# =============================================================================
variables {
  gcp_project_id = "test-project-id"
  gcp_region     = "us-central1"
  environment    = "dev"
  project_name   = "wrinklefree-test"
  ssh_public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest test@example.com"
}

# =============================================================================
# Variable Validation Tests
# =============================================================================

run "valid_environment_dev" {
  command = plan

  variables {
    environment = "dev"
  }

  assert {
    condition     = var.environment == "dev"
    error_message = "Environment should accept 'dev'"
  }
}

run "valid_environment_prod" {
  command = plan

  variables {
    environment = "prod"
  }

  assert {
    condition     = var.environment == "prod"
    error_message = "Environment should accept 'prod'"
  }
}

# =============================================================================
# Network Configuration Tests
# =============================================================================

run "network_naming" {
  command = plan

  assert {
    condition     = google_compute_network.inference.name == "wrinklefree-test-dev-network"
    error_message = "Network name should follow naming convention"
  }
}

run "subnet_naming" {
  command = plan

  assert {
    condition     = google_compute_subnetwork.inference.name == "wrinklefree-test-dev-subnet"
    error_message = "Subnet name should follow naming convention"
  }
}

run "subnet_cidr_default" {
  command = plan

  assert {
    condition     = google_compute_subnetwork.inference.ip_cidr_range == "10.200.0.0/24"
    error_message = "Subnet CIDR should default to 10.200.0.0/24"
  }
}

run "subnet_private_google_access" {
  command = plan

  assert {
    condition     = google_compute_subnetwork.inference.private_ip_google_access == true
    error_message = "Subnet should have private Google access enabled"
  }
}

# =============================================================================
# Firewall Tests
# =============================================================================

run "firewall_ssh_exists" {
  command = plan

  assert {
    condition     = length(google_compute_firewall.allow_ssh.allow) > 0
    error_message = "SSH firewall rule should exist"
  }

  assert {
    condition     = contains(google_compute_firewall.allow_ssh.allow[0].ports, "22")
    error_message = "SSH firewall should allow port 22"
  }
}

run "firewall_inference_exists" {
  command = plan

  assert {
    condition     = length(google_compute_firewall.allow_inference.allow) > 0
    error_message = "Inference firewall rule should exist"
  }

  assert {
    condition     = contains(google_compute_firewall.allow_inference.allow[0].ports, "8080")
    error_message = "Inference firewall should allow port 8080"
  }
}

run "firewall_health_check_exists" {
  command = plan

  assert {
    condition     = length(google_compute_firewall.allow_health_check.source_ranges) == 2
    error_message = "Health check firewall should allow GCP health check ranges"
  }
}
