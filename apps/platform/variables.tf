variable "project_id" {
  type        = string
  description = "This is the GCP project ID to deploy into"
}
variable "region" {
  type        = string
  description = "This is the GCP region to deploy into"
}
variable "service_account_name" {
  description = "GCP service account"
}
variable "image_uri" {
  description = "Image location in the Container Registry"
}