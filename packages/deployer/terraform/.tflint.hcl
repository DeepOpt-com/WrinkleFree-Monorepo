config {
  format = "compact"
  module = true
}

plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

plugin "aws" {
  enabled = true
  version = "0.30.0"
  source  = "github.com/terraform-linters/tflint-ruleset-aws"
}

plugin "google" {
  enabled = true
  version = "0.28.0"
  source  = "github.com/terraform-linters/tflint-ruleset-google"
}

# Naming convention enforcement
rule "terraform_naming_convention" {
  enabled = true
  format  = "snake_case"
}

# Require variable descriptions
rule "terraform_documented_variables" {
  enabled = true
}

# Require output descriptions
rule "terraform_documented_outputs" {
  enabled = true
}

# Prevent deprecated syntax
rule "terraform_deprecated_interpolation" {
  enabled = true
}

# Enforce standard module structure
rule "terraform_standard_module_structure" {
  enabled = true
}

# GCP-specific rules
rule "google_compute_instance_invalid_machine_type" {
  enabled = true
}
