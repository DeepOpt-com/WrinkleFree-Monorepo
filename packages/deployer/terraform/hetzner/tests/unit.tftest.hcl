# Terraform native tests for Hetzner module
# Run with: terraform test

# =============================================================================
# Test Variables
# =============================================================================
variables {
  hetzner_cloud_token = "test-token-for-testing"
  environment         = "dev"
  project_name        = "wrinklefree-test"
  node_count          = 0
  ssh_public_key_path = "./tests/setup/test_key.pub"
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

run "valid_environment_staging" {
  command = plan

  variables {
    environment = "staging"
  }

  assert {
    condition     = var.environment == "staging"
    error_message = "Environment should accept 'staging'"
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

run "valid_location_fsn1" {
  command = plan

  variables {
    location = "fsn1"
  }

  assert {
    condition     = var.location == "fsn1"
    error_message = "Location should accept 'fsn1'"
  }
}

run "valid_location_nbg1" {
  command = plan

  variables {
    location = "nbg1"
  }

  assert {
    condition     = var.location == "nbg1"
    error_message = "Location should accept 'nbg1'"
  }
}

run "valid_location_hel1" {
  command = plan

  variables {
    location = "hel1"
  }

  assert {
    condition     = var.location == "hel1"
    error_message = "Location should accept 'hel1'"
  }
}

# =============================================================================
# Resource Naming Convention Tests
# =============================================================================

run "resource_naming_convention" {
  command = plan

  assert {
    condition     = hcloud_ssh_key.deploy.name == "wrinklefree-test-dev-deploy"
    error_message = "SSH key name should follow naming convention: project-environment-deploy"
  }

  assert {
    condition     = hcloud_network.inference.name == "wrinklefree-test-dev-network"
    error_message = "Network name should follow naming convention: project-environment-network"
  }

  assert {
    condition     = hcloud_firewall.inference.name == "wrinklefree-test-dev-inference-fw"
    error_message = "Firewall name should follow naming convention: project-environment-inference-fw"
  }
}

# =============================================================================
# Label Tests
# =============================================================================

run "labels_applied_to_network" {
  command = plan

  assert {
    condition     = hcloud_network.inference.labels["project"] == "wrinklefree-test"
    error_message = "Network should have 'project' label"
  }

  assert {
    condition     = hcloud_network.inference.labels["environment"] == "dev"
    error_message = "Network should have 'environment' label"
  }

  assert {
    condition     = hcloud_network.inference.labels["managed_by"] == "terraform"
    error_message = "Network should have 'managed_by' label"
  }
}

run "labels_applied_to_firewall" {
  command = plan

  assert {
    condition     = hcloud_firewall.inference.labels["project"] == "wrinklefree-test"
    error_message = "Firewall should have 'project' label"
  }

  assert {
    condition     = hcloud_firewall.inference.labels["environment"] == "dev"
    error_message = "Firewall should have 'environment' label"
  }
}

# =============================================================================
# Network Configuration Tests
# =============================================================================

run "network_cidr_default" {
  command = plan

  assert {
    condition     = hcloud_network.inference.ip_range == "10.0.0.0/16"
    error_message = "Default network CIDR should be 10.0.0.0/16"
  }

  assert {
    condition     = hcloud_network_subnet.inference.ip_range == "10.0.1.0/24"
    error_message = "Default subnet CIDR should be 10.0.1.0/24"
  }
}

run "subnet_zone_correct" {
  command = plan

  assert {
    condition     = hcloud_network_subnet.inference.network_zone == "eu-central"
    error_message = "Subnet network zone should be eu-central"
  }
}

# =============================================================================
# Server Count Tests
# =============================================================================

run "no_servers_when_count_zero" {
  command = plan

  variables {
    node_count = 0
  }

  assert {
    condition     = length(hcloud_server.inference) == 0
    error_message = "No servers should be created when node_count is 0"
  }
}

run "servers_created_when_count_positive" {
  command = plan

  variables {
    node_count = 2
  }

  assert {
    condition     = length(hcloud_server.inference) == 2
    error_message = "2 servers should be created when node_count is 2"
  }
}

# =============================================================================
# Firewall Configuration Tests
# =============================================================================

run "firewall_has_ssh_rule" {
  command = plan

  assert {
    condition     = length([for r in hcloud_firewall.inference.rule : r if r.port == "22"]) > 0
    error_message = "Firewall should have SSH rule on port 22"
  }
}

run "firewall_has_inference_rules" {
  command = plan

  assert {
    condition     = length([for r in hcloud_firewall.inference.rule : r if r.port == "8080"]) > 0
    error_message = "Firewall should have inference API rules on port 8080"
  }
}

run "firewall_has_icmp_rule" {
  command = plan

  assert {
    condition     = length([for r in hcloud_firewall.inference.rule : r if r.protocol == "icmp"]) > 0
    error_message = "Firewall should have ICMP rule for ping"
  }
}
