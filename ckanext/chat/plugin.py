import ckan.plugins as plugins
import ckan.plugins.toolkit as toolkit
from ckan.config.declaration import Declaration, Key
from ckan.lib.plugins import DefaultTranslation

from ckanext.chat import action, api, auth, helpers, views


class ChatPlugin(plugins.SingletonPlugin, DefaultTranslation):
    plugins.implements(plugins.IConfigurer)
    plugins.implements(plugins.IConfigDeclaration)
    plugins.implements(plugins.IBlueprint)
    plugins.implements(plugins.ITemplateHelpers)
    plugins.implements(plugins.IAuthFunctions)
    plugins.implements(plugins.IActions)
    plugins.implements(plugins.ITranslation)

    # IConfigurer

    def update_config(self, config_):
        toolkit.add_template_directory(config_, "templates")
        toolkit.add_public_directory(config_, "public")
        toolkit.add_resource("assets", "chat")

    # IConfigDeclaration

    def declare_config_options(self, declaration: Declaration, key: Key):

        declaration.annotate("chat")
        group = key.ckanext.chat
        declaration.declare_bool(group.ssl_verify, True)
        declaration.declare(group.completion_url, "https://your.chat.api")
        declaration.declare(group.deployment, "gpt-4o-mini")
        declaration.declare(group.api_token, "your-api-token")
        declaration.declare(group.embedding_model, "text-embedding-3-small")
        declaration.declare(group.embedding_api, "")
        declaration.declare(group.milvus_url, "")
        declaration.declare(group.milvus_token, "")
        declaration.declare(group.collection_name, "")
        declaration.declare(group.provider, "azure")
        declaration.declare(group.model_name, "")
        declaration.declare(group.think_model_name, "")
        declaration.declare(group.base_url, "")
        declaration.declare(group.api_key, "")
        declaration.declare(group.api_version, "2024-06-01")
        declaration.declare_int(group.embedding_timeout, 15)
        declaration.declare(
            group.agent_bypass_actions,
            "organization_list,organization_show,group_list,group_show,tag_list,user_show,resource_show,package_show,package_search,package_list"
        )
        declaration.declare(
            group.agent_required_suffixes,
            "_create,_patch"
        )

    # IBlueprint

    def get_blueprint(self):
        return [views.get_blueprint(), api.get_api_blueprint()]

    # ITemplateHelpers

    def get_helpers(self):
        return helpers.get_helpers()

    # IAuthFunctions

    def get_auth_functions(self):
        return auth.get_auth_functions()

    # IActions

    def get_actions(self):
        actions = action.get_actions()
        return actions
