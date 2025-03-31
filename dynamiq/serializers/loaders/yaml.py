from os import PathLike
from typing import Any

from dynamiq import Workflow
from dynamiq.connections import BaseConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.nodes.managers import NodeManager
from dynamiq.nodes.node import ConnectionNode, NodeDependency
from dynamiq.prompts import Prompt
from dynamiq.serializers.types import WorkflowYamlData
from dynamiq.utils.logger import logger


class WorkflowYAMLLoaderException(Exception):
    """Exception raised for errors in the WorkflowYAMLDumper."""

    pass


class WorkflowYAMLLoader:
    """Loader class for parsing YAML files and creating workflow components."""

    @classmethod
    def get_entity_by_type(cls, entity_type: str, entity_registry: dict[str, Any] | None = None) -> Any:
        """
        Try to get entity by type and update mutable shared registry.

        Args:
            entity_type (str): The type of entity to retrieve.
            entity_registry (dict[str, Any] | None): A registry of entities.

        Returns:
            Any: The retrieved entity.

        Raises:
            WorkflowYAMLLoaderException: If the entity is not valid or cannot be found.
        """
        if entity_registry is None:
            entity_registry = {}

        if entity := entity_registry.get(entity_type):
            return entity

        try:
            entity = ConnectionManager.get_connection_by_type(entity_type)
        except ValueError:
            pass

        if not entity:
            try:
                entity = NodeManager.get_node_by_type(entity_type)
            except ValueError:
                pass

        if not entity:
            raise WorkflowYAMLLoaderException(f"Entity '{entity_type}' is not valid.")

        entity_registry[entity_type] = entity
        return entity

    @classmethod
    def get_connections(cls, data: dict[str, dict], registry: dict[str, Any]) -> dict[str, BaseConnection]:
        """
        Get connections from the provided data.

        Args:
            data (dict[str, dict]): The data containing connection information.
            registry (dict[str, Any]): A registry of entities.

        Returns:
            dict[str, BaseConnection]: A dictionary of connections.

        Raises:
            WorkflowYAMLLoaderException: If there's an error in connection data or initialization.
        """
        connections = {}
        for conn_id, conn_data in data.get("connections", {}).items():
            if conn_id in connections:
                raise WorkflowYAMLLoaderException(f"Connection '{conn_id}' already exists")
            if not (conn_type := conn_data.get("type")):
                raise WorkflowYAMLLoaderException(f"Value 'type' not found for connection '{conn_id}'")

            conn_cls = cls.get_entity_by_type(entity_type=conn_type, entity_registry=registry)
            conn_init_data = conn_data | {"id": conn_id}
            conn_init_data.pop("type", None)
            try:
                connection = conn_cls(**conn_init_data)
            except Exception as e:
                raise WorkflowYAMLLoaderException(
                    f"Connection '{conn_id}' data is invalid. Data: '{conn_data}'. Error: {e}"
                )

            connections[conn_id] = connection

        return connections

    @classmethod
    def init_prompt(cls, prompt_init_data: dict) -> Prompt:
        """
        Initialize a prompt from the provided data.

        Args:
            prompt_init_data (dict): The data for the prompt.

        Returns:
            Prompt: The initialized prompt.

        Raises:
            WorkflowYAMLLoaderException: If the specified prompt is not found.
        """
        try:
            return Prompt(**prompt_init_data)
        except Exception as e:
            raise WorkflowYAMLLoaderException(f"Prompt data is invalid. Data: {prompt_init_data}. " f"Error: {e}")

    @classmethod
    def get_prompts(cls, data: dict[str, dict]) -> dict[str, Prompt]:
        """
        Get prompts from the provided data.

        Args:
            data (dict[str, dict]): The data containing prompt information.

        Returns:
            dict[str, Prompt]: A dictionary of prompts.

        Raises:
            WorkflowYAMLLoaderException: If there's an error in prompt data or initialization.
        """
        prompts = {}
        for prompt_id, prompt_data in data.get("prompts", {}).items():
            if prompt_id in prompts:
                raise WorkflowYAMLLoaderException(f"Prompt '{prompt_id}' already exists")
            prompts[prompt_id] = cls.init_prompt(prompt_data | {"id": prompt_id})

        return prompts

    @classmethod
    def get_node_prompt(cls, node_id: str, node_data: dict, prompts: dict[id, Prompt]) -> Prompt | None:
        """
        Get the prompt for a node.

        Args:
            node_id (str): The ID of the node.
            node_data (dict): The data for the node.
            prompts (dict[id, Prompt]): A dictionary of available prompts.

        Returns:
            Prompt | None: The prompt for the node, or None if not found.

        Raises:
            WorkflowYAMLLoaderException: If the specified prompt is not found.
        """
        prompt = None
        if prompt_id := node_data.get("prompt"):
            prompt = prompts.get(prompt_id)
            if not prompt:
                raise WorkflowYAMLLoaderException(f"Prompt '{prompt_id}' for node '{node_id}' not found")
        return prompt

    @classmethod
    def get_node_connection(
        cls, node_id: str, node_data: dict, connections: dict[id, BaseConnection]
    ) -> BaseConnection | None:
        """
        Get the connection for a node.

        Args:
            node_id (str): The ID of the node.
            node_data (dict): The data for the node.
            connections (dict[id, BaseConnection]): A dictionary of available connections.

        Returns:
            BaseConnection | None: The connection for the node, or None if not found.

        Raises:
            WorkflowYAMLLoaderException: If the specified connection is not found.
        """
        conn = None
        if conn_id := node_data.get("connection"):
            conn = connections.get(conn_id)
            if not conn:
                raise WorkflowYAMLLoaderException(f"Connection '{conn_id}' for node '{node_id}' not found")
        return conn

    @classmethod
    def get_node_vector_store_connection(
        cls, node_id: str, node_data: dict, connections: dict[id, BaseConnection]
    ) -> Any | None:
        """
        Get the vector store connection for a node.

        Args:
            node_id (str): The ID of the node.
            node_data (dict): The data for the node.
            connections (dict[id, BaseConnection]): A dictionary of available connections.

        Returns:
            Any | None: The vector store connection for the node, or None if not found.

        Raises:
            WorkflowYAMLLoaderException: If the specified vector store connection is not found or
                                         does not support vector store initialization.
        """
        if conn := cls.get_node_connection(node_id=node_id, node_data=node_data, connections=connections):
            if not (conn_to_vs := getattr(conn, "connect_to_vector_store", None)) or not callable(conn_to_vs):
                raise WorkflowYAMLLoaderException(
                    f"Vector store connection '{conn.id}' for node '{node_id}' not support vector store initialization"
                )
        return conn

    @classmethod
    def get_node_flow(cls, node_id: str, node_data: dict, flows: dict[id, Flow]) -> Flow | None:
        """
        Get the flow for a node.

        Args:
            node_id (str): The ID of the node.
            node_data (dict): The data for the node.
            flows (dict[id, Flow]): A dictionary of available flows.

        Returns:
            Flow | None: The flow for the node, or None if not found.

        Raises:
            WorkflowYAMLLoaderException: If the specified flow is not found.
        """
        flow = None
        if flow_id := node_data.get("flow"):
            flow = flows.get(flow_id)
            if not flow:
                raise WorkflowYAMLLoaderException(f"Flow '{flow_id}' for node '{node_id}' not found")
        return flow

    @classmethod
    def get_node_flows(cls, node_id: str, node_data: dict, flows: dict[id, Flow]) -> list[Flow]:
        """
        Get the flows for a node.

        Args:
            node_id (str): The ID of the node.
            node_data (dict): The data for the node.
            flows (dict[id, Flow]): A dictionary of available flows.

        Returns:
            list[Flow]: A list of flows for the node.

        Raises:
            WorkflowYAMLLoaderException: If any specified flow is not found.
        """
        node_flows = []
        for flow_id in node_data.get("flows", []):
            node_flow = flows.get(flow_id)
            if not node_flow:
                raise WorkflowYAMLLoaderException(f"Flow '{flow_id}' for node '{node_id}' not found")
            node_flows.append(node_flow)
        return node_flows

    @classmethod
    def get_node_dependencies(cls, node_id: str, node_data: dict, nodes: dict[str, Node]):
        """
        Get the dependencies for a node.

        Args:
            node_id (str): The ID of the node.
            node_data (dict): The data for the node.
            nodes (dict[str, Node]): A dictionary of available nodes.

        Returns:
            list[NodeDependency]: A list of node dependencies.

        Raises:
            WorkflowYAMLLoaderException: If there's an error in dependency data or initialization.
        """
        node_depends = []
        for dependency_data in node_data.get("depends", []):
            dependency_node = nodes.get(dependency_data.get("node"))
            dependency_init_data = dependency_data | {"node": dependency_node}
            try:
                dependency = NodeDependency(**dependency_init_data)
            except Exception as e:
                raise WorkflowYAMLLoaderException(
                    f"Dependency data for node '{node_id}' is invalid. Data: {dependency_data}. Error: {e}"
                )

            if dependency.option:
                if not (dep_options := getattr(dependency_node, "options", [])):
                    raise WorkflowYAMLLoaderException(
                        f"Dependency '{dependency.node}' with option '{dependency.option}' "
                        f"for node '{node_id}' not found"
                    )

                if not any(opt.id == dependency.option for opt in dep_options):
                    raise WorkflowYAMLLoaderException(
                        f"Dependency '{dependency.node}' with option '{dependency.option}' "
                        f"for node '{node_id}' not found"
                    )

            node_depends.append(dependency)
        return node_depends

    @classmethod
    def get_updated_node_init_data_with_initialized_nodes(
        cls,
        node_init_data: dict,
        nodes: dict[str, Node],
        flows: dict[str, Flow],
        connections: dict[str, BaseConnection],
        prompts: dict[str, Prompt],
        registry: dict[str, Any],
        connection_manager: ConnectionManager | None = None,
        init_components: bool = False,
    ):
        """
        Get node init data with initialized nodes components recursively (llms, agents, etc)

        Args:
            node_init_data: Dictionary containing node data.
            nodes: Existing nodes dictionary.
            flows: Existing flows dictionary.
            connections: Existing connections dictionary.
            prompts: Existing prompts dictionary.
            registry: Registry of node types.
            connection_manager: Optional connection manager.
            init_components: Flag to initialize components.

        Returns:
            A dictionary of newly created nodes with dependencies.
        """
        updated_node_init_data = {}
        kwargs = dict(
            nodes=nodes,
            flows=flows,
            connections=connections,
            prompts=prompts,
            registry=registry,
            connection_manager=connection_manager,
            init_components=init_components,
        )
        for param_name, param_data in node_init_data.items():
            # TODO: dummy fix, revisit this!
            # We had to add this condition because both input and output nodes have a `schema` param,
            # which has a `type` field that contains types supported by JSON schema (e.g., string, object).
            if param_name == "schema":
                updated_node_init_data[param_name] = param_data

            elif isinstance(param_data, dict):
                updated_param_data = {}
                for param_name_inner, param_data_inner in param_data.items():
                    if param_name_inner == "prompt":
                        updated_param_data[param_name_inner] = param_data_inner
                    elif isinstance(param_data_inner, (dict, list)):
                        param_id = None
                        updated_param_data[param_name_inner] = cls.get_updated_node_init_data_with_initialized_nodes(
                            {param_id: param_data_inner}, **kwargs
                        )[param_id]
                    else:
                        updated_param_data[param_name_inner] = param_data_inner

                if "type" in updated_param_data:
                    param_id = updated_param_data.get("id")
                    updated_param_data = cls.get_nodes_without_depends({param_id: updated_param_data}, **kwargs)[
                        param_id
                    ]

                updated_node_init_data[param_name] = updated_param_data

            elif isinstance(param_data, list):
                updated_items = []
                for item in param_data:
                    if isinstance(item, (dict, list)):
                        param_id = None
                        updated_items.append(
                            cls.get_updated_node_init_data_with_initialized_nodes(
                                node_init_data={param_id: item}, **kwargs
                            )[param_id]
                        )
                    else:
                        updated_items.append(item)
                updated_node_init_data[param_name] = updated_items

            else:
                updated_node_init_data[param_name] = param_data

        return updated_node_init_data

    @classmethod
    def get_nodes_without_depends(
        cls,
        data: dict,
        nodes: dict[str, Node],
        flows: dict[str, Flow],
        connections: dict[str, BaseConnection],
        prompts: dict[str, Prompt],
        registry: dict[str, Any],
        connection_manager: ConnectionManager | None = None,
        init_components: bool = False,
    ) -> dict[str, Node]:
        """
        Create nodes without dependencies from the given data.

        Args:
            data: Dictionary containing node data.
            nodes: Existing nodes dictionary.
            flows: Existing flows dictionary.
            connections: Existing connections dictionary.
            prompts: Existing prompts dictionary.
            registry: Registry of node types.
            connection_manager: Optional connection manager.
            init_components: Flag to initialize components.

        Returns:
            A dictionary of newly created nodes without dependencies.

        Raises:
            WorkflowYAMLLoaderException: If node data is invalid or duplicates are found.
        """
        new_nodes = {}
        for node_id, node_data in data.items():
            if node_id in nodes:
                continue

            if node_id in new_nodes:
                raise WorkflowYAMLLoaderException(f"Node '{node_id}' already exists")

            if not (node_type := node_data.get("type")):
                raise WorkflowYAMLLoaderException(f"Value 'type' for node '{node_id}' not found")

            node_cls = cls.get_entity_by_type(entity_type=node_type, entity_registry=registry)

            # Init node params
            node_init_data = node_data.copy()
            if node_id:
                node_init_data["id"] = node_id
            node_init_data.pop("type", None)
            node_init_data.pop("depends", None)

            if "is_postponed_component_init" not in node_init_data:
                node_init_data["is_postponed_component_init"] = True

            if "connection" in node_init_data:
                get_node_conn = (
                    cls.get_node_vector_store_connection
                    if isinstance(node_cls, ConnectionNode)
                    else cls.get_node_connection
                )
                node_init_data["connection"] = get_node_conn(
                    node_id=node_id, node_data=node_data, connections=connections
                )
            if prompt_data := node_init_data.get("prompt"):
                node_init_data["prompt"] = (
                    cls.get_node_prompt(node_id=node_id, node_data=node_data, prompts=prompts)
                    if isinstance(prompt_data, str)
                    else cls.init_prompt(prompt_data)
                )
            if "flow" in node_init_data:
                node_init_data["flow"] = cls.get_node_flow(node_id=node_id, node_data=node_data, flows=flows)
            if "flows" in node_init_data:
                node_init_data["flows"] = cls.get_node_flows(node_id=node_id, node_data=node_data, flows=flows)
            try:

                node_init_data = cls.get_updated_node_init_data_with_initialized_nodes(
                    node_init_data=node_init_data,
                    nodes=nodes,
                    flows=flows,
                    connections=connections,
                    prompts=prompts,
                    registry=registry,
                    connection_manager=connection_manager,
                    init_components=init_components,
                )

                node = node_cls(**node_init_data)

                if init_components and getattr(node, "init_components", False):
                    node.init_components(connection_manager=connection_manager)
                    node.is_postponed_component_init = False

            except Exception as e:
                raise WorkflowYAMLLoaderException(f"Node '{node_id}' data is invalid. Data: {node_data}. Error: {e}")

            new_nodes[node_id] = node

        return new_nodes

    @classmethod
    def get_nodes(
        cls,
        nodes_data: dict,
        nodes: dict[str, Node],
        flows: dict[str, Flow],
        connections: dict[str, BaseConnection],
        prompts: dict[str, Prompt],
        registry: dict[str, Any],
        connection_manager: ConnectionManager | None = None,
        init_components: bool = False,
    ):
        """
        Create nodes with dependencies from the given data.

        Args:
            nodes_data: Dictionary containing node data.
            nodes: Existing nodes dictionary.
            flows: Existing flows dictionary.
            connections: Existing connections dictionary.
            prompts: Existing prompts dictionary.
            registry: Registry of node types.
            connection_manager: Optional connection manager.
            init_components: Flag to initialize components.

        Returns:
            A dictionary of newly created nodes with dependencies.
        """

        new_nodes = cls.get_nodes_without_depends(
            data=nodes_data,
            nodes=nodes,
            flows=flows,
            connections=connections,
            prompts=prompts,
            registry=registry,
            connection_manager=connection_manager,
            init_components=init_components,
        )

        all_nodes = nodes | new_nodes
        for node_id, node in new_nodes.items():
            node.depends = cls.get_node_dependencies(node_id=node_id, node_data=nodes_data[node_id], nodes=all_nodes)

        return new_nodes

    @classmethod
    def get_dependant_nodes(
        cls,
        nodes_data: dict[str, dict],
        flows_data: dict[str, dict],
        connections: dict[str, BaseConnection],
        prompts: dict[str, Prompt],
        registry: dict[str, Any],
        connection_manager: ConnectionManager | None = None,
        init_components: bool = False,
    ) -> dict[str, Node]:
        """
        Get nodes that are dependent on flows.

        Args:
            nodes_data: Dictionary containing node data.
            flows_data: Dictionary containing flow data.
            connections: Existing connections dictionary.
            prompts: Existing prompts dictionary.
            registry: Registry of node types.
            connection_manager: Optional connection manager.
            init_components: Flag to initialize components.

        Returns:
            A dictionary of nodes that are dependent on flows.
        """
        dependant_nodes, dependant_nodes_data = {}, {}
        dependant_flow_ids = []

        for node_id, node_data in nodes_data.items():
            if "flow" in node_data:
                dependant_nodes_data[node_id] = node_data
                dependant_flow_ids.append(node_data["flow"])
            if "flows" in node_data:
                dependant_nodes_data[node_id] = node_data
                dependant_flow_ids.extend(node_data["flows"])

        # Get nodes from dependant flows
        if dependant_flow_ids:
            dependant_flows_nodes_ids = []
            for flow_id, flow_data in flows_data.items():
                if flow_id in dependant_flow_ids:
                    dependant_flows_nodes_ids.extend(flow_data.get("nodes", []))

            dependant_flows_nodes_data = {
                node_id: node_data for node_id, node_data in nodes_data.items() if node_id in dependant_flows_nodes_ids
            }

            dependant_nodes = cls.get_nodes(
                nodes_data=dependant_flows_nodes_data,
                nodes={},
                flows={},
                connections=connections,
                prompts=prompts,
                registry=registry,
                connection_manager=connection_manager,
                init_components=init_components,
            )

        return dependant_nodes

    @classmethod
    def get_flows(
        cls,
        data: dict,
        flows: dict[str, Flow],
        nodes: dict[str, Node],
        connection_manager: ConnectionManager | None = None,
    ) -> dict[str, Flow]:
        """
        Create flows from the given data.

        Args:
            data: Dictionary containing flow data.
            flows: Existing flows dictionary.
            nodes: Existing nodes dictionary.
            connection_manager: Optional connection manager.

        Returns:
            A dictionary of newly created flows.

        Raises:
            WorkflowYAMLLoaderException: If flow data is invalid or duplicates are found.
        """
        new_flows = {}
        for flow_id, flow_data in data.items():
            if flow_id in flows:
                continue

            if flow_id in new_flows:
                raise WorkflowYAMLLoaderException(f"Flow {flow_id} already exists")

            flow_node_ids = flow_data.get("nodes", [])
            flow_node_ids = set(flow_node_ids)
            dep_node_ids = set()
            for node_id in flow_node_ids:
                if node_id not in nodes:
                    raise WorkflowYAMLLoaderException(f"Node '{node_id}' for flow '{flow_id}' not found")

                dep_node_ids.update({dep.node.id for dep in nodes[node_id].depends})

            for node_id in dep_node_ids:
                if node_id not in flow_node_ids:
                    raise WorkflowYAMLLoaderException(
                        f"Dependency node '{node_id}' in the flow '{flow_id}' node list not found"
                    )

            flow_init_data = flow_data | {
                "id": flow_id,
                "nodes": [nodes[node_id] for node_id in flow_node_ids],
            }
            if connection_manager:
                flow_init_data["connection_manager"] = connection_manager

            try:
                flow = Flow(**flow_init_data)
            except Exception as e:
                raise WorkflowYAMLLoaderException(f"Flow '{flow_id}' data is invalid. Data: {flow_data}. Error: {e}")

            new_flows[flow_id] = flow
        return new_flows

    @classmethod
    def get_dependant_flows(
        cls,
        nodes_data: dict[str, dict],
        flows_data: dict[str, dict],
        dependant_nodes: dict[str, Node],
        connection_manager: ConnectionManager | None = None,
    ) -> dict[str, Flow]:
        """
        Get flows that are dependent on nodes.

        Args:
            nodes_data: Dictionary containing node data.
            flows_data: Dictionary containing flow data.
            dependant_nodes: Dictionary of dependent nodes.
            connection_manager: Optional connection manager.

        Returns:
            A dictionary of flows that are dependent on nodes.
        """
        dependant_flows = {}
        dependant_flow_ids = []

        for node_id, node_data in nodes_data.items():
            if "flow" in node_data:
                dependant_flow_ids.append(node_data["flow"])
            if "flows" in node_data:
                dependant_flow_ids.extend(node_data["flows"])

        if dependant_flow_ids:
            dependant_flows_data = {
                flow_id: flow_data for flow_id, flow_data in flows_data.items() if flow_id in dependant_flow_ids
            }
            dependant_flows = cls.get_flows(
                data=dependant_flows_data,
                flows={},
                nodes=dependant_nodes,
                connection_manager=connection_manager,
            )

        return dependant_flows

    @classmethod
    def get_workflows(cls, data: dict, flows: dict[str, Flow]) -> dict[str, Workflow]:
        """
        Create workflows from the given data.

        Args:
            data: Dictionary containing workflow data.
            flows: Existing flows dictionary.

        Returns:
            A dictionary of newly created workflows.

        Raises:
            WorkflowYAMLLoaderException: If workflow data is invalid.
        """
        workflows = {}
        for wf_id, wf_data in data.get("workflows", {}).items():
            if not (flow_id := wf_data.get("flow")):
                raise WorkflowYAMLLoaderException(f"Value 'flow' for dynamiq '{wf_id}' not found ")
            if not (flow := flows.get(flow_id)):
                raise WorkflowYAMLLoaderException(f"Flow '{flow_id}' for dynamiq '{wf_id}' not found")
            if version := wf_data.get("version"):
                version = str(version)

            try:
                wf = Workflow(id=wf_id, flow=flow, version=version)
            except Exception as e:
                raise WorkflowYAMLLoaderException(f"Workflow '{wf_id}' data is invalid. Data: {wf_data}. Error: {e}")

            workflows[wf_id] = wf
        return workflows

    @classmethod
    def load(
        cls,
        file_path: str | PathLike,
        connection_manager: ConnectionManager | None = None,
        init_components: bool = False,
    ) -> WorkflowYamlData:
        """
        Load data from a YAML file and parse it.

        Args:
            file_path: Path to the YAML file.
            connection_manager: Optional connection manager.
            init_components: Flag to initialize components.

        Returns:
            Parsed WorkflowYamlData object.
        """
        data = cls.loads(file_path)
        return cls.parse(
            data=data,
            connection_manager=connection_manager,
            init_components=init_components,
        )

    @classmethod
    def loads(cls, file_path: str | PathLike):
        """
        Load data from a YAML file.

        Args:
            file_path: Path to the YAML file.

        Returns:
            Parsed data from the YAML file.

        Raises:
            WorkflowYAMLLoaderException: If the file is not found.
        """
        from omegaconf import OmegaConf

        try:
            conf = OmegaConf.load(file_path)
            logger.debug(f"Loaded config from '{file_path}'")

            data = OmegaConf.to_container(conf, resolve=True)
        except FileNotFoundError:
            raise WorkflowYAMLLoaderException(f"File '{file_path}' not found")

        return data

    @classmethod
    def parse(
        cls,
        data: dict,
        connection_manager: ConnectionManager | None = None,
        init_components: bool = False,
    ) -> WorkflowYamlData:
        """
        Parse dynamiq workflow data.

        Args:
            data: Dictionary containing workflow data.
            connection_manager: Optional connection manager.
            init_components: Flag to initialize components.

        Returns:
            Parsed WorkflowYamlData object.

        Raises:
            WorkflowYAMLLoaderException: If parsing fails.
        """
        nodes, flows = {}, {}
        # Mutable shared registry that updates with each new entity.
        node_registry, connection_registry = {}, {}
        if init_components and connection_manager is None:
            connection_manager = ConnectionManager()

        try:
            connections = cls.get_connections(data=data, registry=connection_registry)
            prompts = cls.get_prompts(data)

            nodes_data = data.get("nodes", {})
            flows_data = data.get("flows", {})

            dependant_nodes = cls.get_dependant_nodes(
                nodes_data=nodes_data,
                flows_data=flows_data,
                connections=connections,
                prompts=prompts,
                registry=node_registry,
                connection_manager=connection_manager,
                init_components=init_components,
            )
            nodes.update(dependant_nodes)

            dependant_flows = cls.get_dependant_flows(
                nodes_data=nodes_data,
                flows_data=flows_data,
                dependant_nodes=dependant_nodes,
                connection_manager=connection_manager,
            )
            flows.update(dependant_flows)

            non_dependant_nodes = cls.get_nodes(
                nodes_data=nodes_data,
                nodes=nodes,
                flows=flows,
                connections=connections,
                prompts=prompts,
                registry=node_registry,
                connection_manager=connection_manager,
                init_components=init_components,
            )
            nodes.update(non_dependant_nodes)

            non_dependant_flows = cls.get_flows(
                data=flows_data,
                flows=flows,
                nodes=nodes,
                connection_manager=connection_manager,
            )
            flows.update(non_dependant_flows)

            workflows = cls.get_workflows(data, flows)
        except WorkflowYAMLLoaderException:
            raise
        except Exception:
            logger.exception("Failed to parse Yaml data with unexpected error")
            raise

        return WorkflowYamlData(
            connections=connections,
            nodes=nodes,
            flows=flows,
            workflows=workflows,
        )
