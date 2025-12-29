# Cost optimization policies for WrinkleFree infrastructure
# Run with: conftest test terraform/*/tfplan.json --policy terraform/policies/

package terraform.cost

import future.keywords.in

# Warn on expensive Hetzner server types
warn[msg] {
    resource := input.resource_changes[_]
    resource.type == "hcloud_server"
    expensive_types := {"cx91", "ccx63", "cpx51"}
    resource.change.after.server_type in expensive_types
    msg := sprintf("Server '%s' uses expensive type '%s'. Consider if this is necessary.", [
        resource.change.after.name,
        resource.change.after.server_type
    ])
}

# Ensure GCP instances use spot/preemptible in production
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "google_compute_instance_template"
    resource.change.after.labels.environment == "prod"
    scheduling := resource.change.after.scheduling[0]
    scheduling.preemptible != true
    msg := sprintf("GCP instance template '%s' should use preemptible instances in production for cost savings", [
        resource.address
    ])
}

# Ensure GCP uses SPOT provisioning model
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "google_compute_instance_template"
    resource.change.after.labels.environment == "prod"
    scheduling := resource.change.after.scheduling[0]
    scheduling.provisioning_model != "SPOT"
    msg := sprintf("GCP instance template '%s' should use SPOT provisioning model", [
        resource.address
    ])
}

# Warn on autoscaler with high max replicas
warn[msg] {
    resource := input.resource_changes[_]
    resource.type == "google_compute_region_autoscaler"
    policy := resource.change.after.autoscaling_policy[0]
    policy.max_replicas > 20
    msg := sprintf("Autoscaler '%s' has max_replicas=%d which could lead to high costs", [
        resource.address,
        policy.max_replicas
    ])
}

# Ensure autoscaler allows scale-to-zero
warn[msg] {
    resource := input.resource_changes[_]
    resource.type == "google_compute_region_autoscaler"
    policy := resource.change.after.autoscaling_policy[0]
    policy.min_replicas > 0
    resource.change.after.name
    contains(resource.change.after.name, "spot")
    msg := sprintf("Autoscaler '%s' has min_replicas=%d. Consider scale-to-zero for cost savings.", [
        resource.address,
        policy.min_replicas
    ])
}

# Warn on large disk sizes
warn[msg] {
    resource := input.resource_changes[_]
    resource.type == "google_compute_instance_template"
    disk := resource.change.after.disk[_]
    disk.disk_size_gb > 200
    msg := sprintf("Instance template '%s' has disk size %dGB which may be excessive", [
        resource.address,
        disk.disk_size_gb
    ])
}
