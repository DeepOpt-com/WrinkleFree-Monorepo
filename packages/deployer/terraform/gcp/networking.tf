# =============================================================================
# VPC Network
# =============================================================================
resource "google_compute_network" "inference" {
  name                    = "${var.project_name}-${var.environment}-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "inference" {
  name          = "${var.project_name}-${var.environment}-subnet"
  ip_cidr_range = var.network_cidr
  region        = var.gcp_region
  network       = google_compute_network.inference.id

  private_ip_google_access = true
}

# =============================================================================
# Firewall Rules
# =============================================================================
resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.project_name}-${var.environment}-allow-ssh"
  network = google_compute_network.inference.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # Consider restricting this in production via IAP
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["${var.project_name}-inference"]
}

resource "google_compute_firewall" "allow_inference" {
  name    = "${var.project_name}-${var.environment}-allow-inference"
  network = google_compute_network.inference.name

  allow {
    protocol = "tcp"
    ports    = [tostring(var.inference_port)]
  }

  # Cloudflare IP ranges
  source_ranges = var.cloudflare_ips
  target_tags   = ["${var.project_name}-inference"]
}

resource "google_compute_firewall" "allow_health_check" {
  name    = "${var.project_name}-${var.environment}-allow-health"
  network = google_compute_network.inference.name

  allow {
    protocol = "tcp"
    ports    = [tostring(var.inference_port)]
  }

  # GCP health check ranges
  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]

  target_tags = ["${var.project_name}-inference"]
}

resource "google_compute_firewall" "allow_internal" {
  name    = "${var.project_name}-${var.environment}-allow-internal"
  network = google_compute_network.inference.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [var.network_cidr]
  target_tags   = ["${var.project_name}-inference"]
}
