import enum
from os import PathLike
from typing import IO, Any

from dynamiq.connections import BaseConnection
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.serializers.types import WorkflowYamlData
from dynamiq.utils.logger import logger


class WorkflowYAMLDumperException(Exception):
    pass


class WorkflowYAMLDumper:
    """Dumper class for parsing workflow components and save them to YAML."""

    @classmethod
    def get_updated_node_data(
        cls,
        node_data: dict,
        connections_data: dict[str, dict],
        skip_nullable: bool = False,
    ):
        """
        Get node init data with and connections components recursively (llms, agents, etc)

        Args:
            node_data: Dictionary containing node data.
            connections_data: Dictionary containing connections shared data.
            skip_nullable: Skip nullable fields.

        Returns:
            A dictionary of newly created nodes with dependencies.
        """
        updated_node_init_data = {}
        for param_name, param_data in node_data.items():
            if param_name == "depends":
                updated_node_init_data[param_name] = [
                    {"node": dep["node"]["id"], "option": dep["option"]} for dep in param_data
                ]

            elif param_name == "connection":
                param_id = None
                connections_data[param_data["id"]] = cls.get_updated_node_data(
                    node_data={param_id: param_data}, connections_data=connections_data, skip_nullable=True
                )[param_id]
                updated_node_init_data[param_name] = param_data["id"]

            elif isinstance(param_data, dict):
                updated_param_data = {}
                for param_name_inner, param_data_inner in param_data.items():
                    if isinstance(param_data_inner, (dict, list)):
                        param_data_inner = cls.get_updated_node_data(
                            node_data={param_name_inner: param_data_inner}, connections_data=connections_data
                        )[param_name_inner]
                    elif isinstance(param_data_inner, enum.Enum):
                        param_data_inner = param_data_inner.value
                    elif param_data_inner is None and skip_nullable:
                        continue

                    updated_param_data[param_name_inner] = param_data_inner

                updated_node_init_data[param_name] = updated_param_data

            elif isinstance(param_data, list):
                updated_items = []
                for item in param_data:
                    if isinstance(item, (dict, list)):
                        param_id = None
                        item = cls.get_updated_node_data(node_data={param_id: item}, connections_data=connections_data)[
                            param_id
                        ]
                    elif isinstance(item, enum.Enum):
                        item = item.value
                    elif item is None and skip_nullable:
                        continue
                    updated_items.append(item)
                updated_node_init_data[param_name] = updated_items

            else:
                if isinstance(param_data, enum.Enum):
                    param_data = param_data.value
                elif param_data is None and skip_nullable:
                    continue
                updated_node_init_data[param_name] = param_data

        return updated_node_init_data

    @classmethod
    def get_nodes_data_and_connections(
        cls, nodes: dict[str, Node], connections: dict[str, BaseConnection]
    ) -> tuple[dict, dict]:
        """
        Get nodes data prepared for Yaml and connections from the given data.

        Args:
            nodes: Existing nodes dictionary.
            connections: Existing connections dictionary.

        Returns:
            A tuple of parsed nodes and connections
        """
        nodes_data = {}
        connections_data = {conn_id: conn.to_dict() for conn_id, conn in connections.items()}
        for node_id, node in nodes.items():
            node_data = cls.get_updated_node_data(
                node_data=node.to_dict(include_secure_params=True, by_alias=True), connections_data=connections_data
            )
            nodes_data[node_id] = node_data

        return nodes_data, connections_data

    @classmethod
    def get_flows_data(cls, flows: dict[str, Flow]):
        """
        Get flows data prepared for Yaml from the given data.

        Args:
            flows: Existing flows dictionary.

        Returns:
            A dictionary of newly created flows
        """
        flows_data = {}
        for flow_id, flow in flows.items():
            flow_data = flow.to_dict(exclude={"nodes", "executor", "connection_manager"})
            flow_data["nodes"] = [node.id for node in flow.nodes]
            flows_data[flow_id] = flow_data

        return flows_data

    @classmethod
    def dump(
        cls,
        file_path: str | PathLike,
        data: WorkflowYamlData,
    ):
        """
        Parse data from a WorkflowYamlData and save it to YAML file.

        Args:
            file_path: Path to the YAML file.
            data: WorkflowYamlData object.
        """
        data = cls.parse(data=data)
        cls.dumps(file_path=file_path, data=data)

    @classmethod
    def dumps(cls, file_path: str | PathLike | IO[Any], data: dict):
        """
        Load data from a YAML file.

        Args:
            file_path: Path to the YAML file.
            data: Data to dump to the YAML file.

        Raises:
            WorkflowYAMLDumperException: If the file is not found.
        """
        from omegaconf import OmegaConf

        try:
            conf = OmegaConf.create(data)
            logger.debug("Dumped data to config")

            OmegaConf.save(config=conf, f=file_path)
        except FileNotFoundError:
            raise WorkflowYAMLDumperException(f"File '{file_path}' not found")

    @classmethod
    def parse(cls, data: WorkflowYamlData) -> dict:
        """
        Parse dynamiq workflow data.

        Args:
            data: WorkflowYamlData object.

        Returns:
            Parsed dict object.

        Raises:
            WorkflowYAMLDumperException: If parsing fails.
        """

        try:
            nodes, connections = cls.get_nodes_data_and_connections(data.nodes, data.connections)
            flows = cls.get_flows_data(data.flows)
            wf_data = {
                "connections": connections,
                "nodes": nodes,
                "flows": flows,
                "workflows": {
                    wf_id: wf.to_dict(exclude={"flow"}) | {"flow": wf.flow.id} for wf_id, wf in data.workflows.items()
                },
            }
        except Exception as e:
            logger.exception("Failed to parse WorkflowYamlData object with unexpected error")
            raise WorkflowYAMLDumperException from e

        return wf_data
