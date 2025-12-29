# =============================================================================
# Server IPs
# =============================================================================
output "server_ips" {
  description = "Public IPv4 addresses of Hetzner Cloud servers"
  value       = hcloud_server.inference[*].ipv4_address
}

output "server_private_ips" {
  description = "Private IPv4 addresses of Hetzner Cloud servers"
  value       = [for s in hcloud_server.inference : s.network[*].ip]
}

# =============================================================================
# SkyPilot Integration
# =============================================================================
output "ssh_node_pools_config" {
  description = "Ready-to-use SkyPilot SSH Node Pools configuration"
  value = yamlencode({
    "${var.project_name}-${var.environment}" = {
      user          = var.ssh_user
      identity_file = pathexpand(replace(var.ssh_public_key_path, ".pub", ""))
      hosts         = [for ip in local.all_server_ips : { ip = ip }]
    }
  })
}

output "skypilot_config_path" {
  description = "Suggested path for SkyPilot SSH node pools config"
  value       = "~/.sky/ssh_node_pools.yaml"
}

# =============================================================================
# Network
# =============================================================================
output "network_id" {
  description = "Hetzner Cloud network ID"
  value       = hcloud_network.inference.id
}

output "subnet_id" {
  description = "Hetzner Cloud subnet ID"
  value       = hcloud_network_subnet.inference.id
}

output "firewall_id" {
  description = "Hetzner Cloud firewall ID"
  value       = hcloud_firewall.inference.id
}

# =============================================================================
# SSH Key
# =============================================================================
output "ssh_key_id" {
  description = "Hetzner Cloud SSH key ID"
  value       = hcloud_ssh_key.deploy.id
}

output "ssh_key_fingerprint" {
  description = "SSH key fingerprint"
  value       = hcloud_ssh_key.deploy.fingerprint
}

# =============================================================================
# For other modules
# =============================================================================
output "for_gcp_module" {
  description = "Outputs formatted for GCP module integration"
  value = {
    hetzner_network_cidr = var.network_cidr
  }
}
