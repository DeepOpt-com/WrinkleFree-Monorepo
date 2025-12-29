locals {
  common_labels = merge(
    {
      project     = var.project_name
      environment = var.environment
      managed_by  = "terraform"
    },
    var.labels
  )

  # All server IPs for SkyPilot integration
  all_server_ips = hcloud_server.inference[*].ipv4_address
}

# =============================================================================
# SSH Key for all servers
# =============================================================================
resource "hcloud_ssh_key" "deploy" {
  name       = "${var.project_name}-${var.environment}-deploy"
  public_key = file(pathexpand(var.ssh_public_key_path))

  labels = local.common_labels
}

# =============================================================================
# Private Network (vSwitch)
# =============================================================================
resource "hcloud_network" "inference" {
  name     = "${var.project_name}-${var.environment}-network"
  ip_range = var.network_cidr

  labels = local.common_labels
}

resource "hcloud_network_subnet" "inference" {
  network_id   = hcloud_network.inference.id
  type         = "cloud"
  network_zone = "eu-central"
  ip_range     = var.subnet_cidr
}

# =============================================================================
# Hetzner Cloud Servers
# =============================================================================
resource "hcloud_server" "inference" {
  count = var.node_count

  name        = "${var.project_name}-${var.environment}-inference-${count.index}"
  server_type = var.server_type
  image       = var.image
  location    = var.location

  ssh_keys = [hcloud_ssh_key.deploy.id]

  labels = merge(local.common_labels, {
    role  = "inference"
    index = tostring(count.index)
  })

  # Attach to private network
  network {
    network_id = hcloud_network.inference.id
    ip         = cidrhost(var.subnet_cidr, count.index + 10)
  }

  # Cloud-init for base setup
  user_data = <<-EOF
    #cloud-config
    package_update: true
    packages:
      - python3
      - python3-pip
      - python3-venv
      - docker.io
    runcmd:
      - systemctl enable docker
      - systemctl start docker
  EOF

  depends_on = [hcloud_network_subnet.inference]
}

# =============================================================================
# Firewall
# =============================================================================
resource "hcloud_firewall" "inference" {
  name = "${var.project_name}-${var.environment}-inference-fw"

  labels = local.common_labels

  # SSH access (restricted)
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = [var.admin_ip]
  }

  # Inference API from Cloudflare
  dynamic "rule" {
    for_each = var.cloudflare_ips
    content {
      direction  = "in"
      protocol   = "tcp"
      port       = tostring(var.inference_port)
      source_ips = [rule.value]
    }
  }

  # Health check from private network
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = tostring(var.inference_port)
    source_ips = [var.network_cidr]
  }

  # ICMP
  rule {
    direction  = "in"
    protocol   = "icmp"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

# Attach firewall to cloud servers
resource "hcloud_firewall_attachment" "inference" {
  count = var.node_count > 0 ? 1 : 0

  firewall_id = hcloud_firewall.inference.id
  server_ids  = hcloud_server.inference[*].id
}
