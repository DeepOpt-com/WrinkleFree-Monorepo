# Security policies for WrinkleFree infrastructure
# Run with: conftest test terraform/*/tfplan.json --policy terraform/policies/

package terraform.security

import future.keywords.in

# Deny SSH access from 0.0.0.0/0 in production
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "hcloud_firewall"
    rule := resource.change.after.rule[_]
    rule.port == "22"
    source := rule.source_ips[_]
    source == "0.0.0.0/0"
    contains(resource.change.after.name, "prod")
    msg := sprintf("Firewall '%s' allows SSH from 0.0.0.0/0 in production", [resource.change.after.name])
}

# Deny GCP firewall with unrestricted SSH in production
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "google_compute_firewall"
    contains(resource.change.after.name, "prod")
    contains(resource.change.after.name, "ssh")
    source := resource.change.after.source_ranges[_]
    source == "0.0.0.0/0"
    msg := sprintf("GCP firewall '%s' allows SSH from 0.0.0.0/0 in production", [resource.change.after.name])
}

# Ensure all resources have required labels (Hetzner)
deny[msg] {
    resource := input.resource_changes[_]
    resource.type in ["hcloud_server", "hcloud_network", "hcloud_firewall"]
    resource.change.actions[_] in ["create", "update"]
    not resource.change.after.labels.project
    msg := sprintf("Resource '%s' is missing required 'project' label", [resource.address])
}

# Ensure all resources have required labels (GCP)
deny[msg] {
    resource := input.resource_changes[_]
    resource.type in ["google_compute_instance_template", "google_compute_network"]
    resource.change.actions[_] in ["create", "update"]
    not resource.change.after.labels.project
    msg := sprintf("Resource '%s' is missing required 'project' label", [resource.address])
}

# Ensure environment label is set
deny[msg] {
    resource := input.resource_changes[_]
    resource.type in ["hcloud_server", "hcloud_network", "google_compute_instance_template"]
    resource.change.actions[_] in ["create", "update"]
    not resource.change.after.labels.environment
    msg := sprintf("Resource '%s' is missing required 'environment' label", [resource.address])
}

# Ensure managed_by label is set (for Terraform tracking)
warn[msg] {
    resource := input.resource_changes[_]
    resource.type in ["hcloud_server", "hcloud_network", "google_compute_instance_template"]
    resource.change.actions[_] in ["create", "update"]
    not resource.change.after.labels.managed_by
    msg := sprintf("Resource '%s' should have 'managed_by' label for tracking", [resource.address])
}
