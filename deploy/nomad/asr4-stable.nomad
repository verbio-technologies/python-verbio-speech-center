variables {
  DOCKER_REGISTRY = "docker.registry.verbio.com/csr"
  VERSION = "latest"
}

variable "envoy_config" {
  type = string
  description = "Custom envoy config"
  default = "envoy.yaml"
}


job "asr4-stable" {
  datacenters = ["dc1"]
  type        = "service"

  meta {
    ASR4_VERSION = "${var.VERSION}"
  }

  group "asr4-group" {
    count = 1

    restart {
      attempts = 10
      interval = "5m"
      delay    = "25s"
      mode     = "delay"
    }

    network {
      port "grpc-port" {
        static = 50061
        to = 10000
      }
      port "grpc-port-en-us" {
        static = 50062
        to = 50051
      }
      port "grpc-port-es" {
        static = 50063
        to = 50051
      }
      port "grpc-port-pt-br" {
        static = 50064
        to = 50051
      }
    }

    task "asr4-en-us-service" {
      driver = "docker"

      config {
        image              = "${var.DOCKER_REGISTRY}/stable/asr4-en-us:${var.VERSION}"
        ports              = ["grpc-port-en-us"]
      }

      logs {
        max_files     = 10
        max_file_size = 10
      }

      env {
        LANGUAGE = "en-us"
      }

      resources {
        memory = 5000
      }

      service {
        name = "asr4-en-us-stable-service"
        port = "grpc-port-en-us"

        check {
          name = "up-and-running"
          type = "grpc"
          port = "grpc-port-en-us"
          interval = "30s"
          timeout = "2s"
        }
      }
    }

    task "asr4-es-service" {
      driver = "docker"

      config {
        image              = "${var.DOCKER_REGISTRY}/stable/asr4-es:${var.VERSION}"
        ports              = ["grpc-port-es"]
      }

      logs {
        max_files     = 10
        max_file_size = 10
      }

      env {
        LANGUAGE = "es"
      }

      resources {
        memory = 5000
      }

      service {
        name = "asr4-es-stable-service"
        port = "grpc-port-es"

        check {
          name = "up-and-running"
          type = "grpc"
          port = "grpc-port-es"
          interval = "30s"
          timeout = "2s"
        }
      }
    }

    task "asr4-pt-br-service" {
      driver = "docker"

      config {
        image              = "${var.DOCKER_REGISTRY}/stable/asr4-pt-br:${var.VERSION}"
        ports              = ["grpc-port-pt-br"]
      }

      logs {
        max_files     = 10
        max_file_size = 10
      }

      env {
        LANGUAGE = "pt-br"
      }

      resources {
        memory = 5000
      }

      service {
        name = "asr4-pt-br-stable-service"
        port = "grpc-port-pt-br"

        check {
          name = "up-and-running"
          type = "grpc"
          port = "grpc-port-pt-br"
          interval = "30s"
          timeout = "2s"
        }
      }
    }

    task "traffic-routing-service" {
      driver = "docker"

      template {
        destination = "tmp/envoy.yaml"
        data = file(var.envoy_config)
      }

      config {
        image              = "${var.DOCKER_REGISTRY}/envoyproxy/envoy:v1.23.1"
        volumes            = ["tmp/envoy.yaml:/etc/envoy/envoy.yaml"]
        ports              = ["grpc-port"]
      }

      service {
        name = "envoy-stable-service"
        port = "grpc-port"
      }
    }
  }
}

